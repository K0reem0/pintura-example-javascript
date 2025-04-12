# -*- coding: utf-8 -*-

import cv2
import numpy as np
import requests
import base64
import time
from PIL import Image, ImageDraw, ImageFont
from requests.exceptions import RequestException
import re
import os
from shapely.geometry import Polygon
from shapely.validation import make_valid

# --- استيراد وحدة التنسيق ---
try:
    import text_formatter
except ImportError:
    print("❌ Error: Could not import 'text_formatter.py'. Make sure it exists in the backend directory.")
    # يمكنك إضافة معالجة خطأ أفضل هنا أو ترك التطبيق يفشل
    raise

# --- الإعدادات العامة ---
# استبدل بمفتاحك الفعلي أو استخدم متغيرات البيئة
ROBOFLOW_TEXT_API_KEY = os.getenv('ROBOFLOW_TEXT_API_KEY', 'tCebtqTr288BJiMWppWM') # مفتاح افتراضي كمثال
ROBOFLOW_BUBBLE_API_KEY = os.getenv('ROBOFLOW_BUBBLE_API_KEY', 'tCebtqTr288BJiMWppWM') # قد يكون نفس المفتاح أو مختلف
LUMINAI_API_URL = "https://luminai.my.id/" # تأكد من صحة هذا الرابط

TEXT_COLOR = (0, 0, 0)          # أسود
SHADOW_COLOR = (255, 255, 255)  # أبيض
SHADOW_OPACITY = 90             # شفافية الظل (0-255)

# --- إعداد الخط ---
# محاولة تحديد مسار الخط العربي وتمريره للمنسق
arabic_font_path_to_set = None
try:
    # البحث في مجلد fonts بنفس مستوى manga_translator.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    potential_path = os.path.join(script_dir, "fonts", "66Hayah.otf")
    if os.path.exists(potential_path):
        arabic_font_path_to_set = potential_path
    else:
        # محاولة البحث في مجلد fonts في المسار الحالي (إذا تم التشغيل من مكان آخر)
        potential_path_cwd = os.path.join(".", "fonts", "66Hayah.otf")
        if os.path.exists(potential_path_cwd):
           arabic_font_path_to_set = potential_path_cwd

    if arabic_font_path_to_set:
        print(f"ℹ️ Backend found Arabic font: {arabic_font_path_to_set}")
        text_formatter.set_arabic_font_path(arabic_font_path_to_set)
    else:
        print("⚠️ Warning: Backend could not find 'fonts/66Hayah.otf'. Using default PIL font via formatter.")
        text_formatter.set_arabic_font_path(None)

except Exception as e:
    print(f"⚠️ Error finding Arabic font: {e}. Using default PIL font via formatter.")
    text_formatter.set_arabic_font_path(None)


# --- دوال مساعدة ---

def decode_image(base64_string):
    """يفك تشفير صورة base64 إلى مصفوفة OpenCV (BGR)."""
    try:
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image data.")
         # ضمان 3 قنوات
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image
    except Exception as e:
        print(f"❌ Error decoding base64 image: {e}")
        return None

def encode_image(image_np, format='.png'):
    """يشفر مصفوفة OpenCV إلى سلسلة base64."""
    try:
        is_success, buffer = cv2.imencode(format, image_np)
        if not is_success:
            raise ValueError("Could not encode image.")
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"❌ Error encoding image to base64: {e}")
        return None

def extract_translation(text):
    """ يستخرج النص الموجود داخل أول علامتي اقتباس مزدوجتين ("...") في السلسلة."""
    if not isinstance(text, str): return ""
    match = re.search(r'"(.*?)"', text, re.DOTALL)
    return match.group(1).strip() if match else text.strip('"').strip()

def ask_luminai(prompt, image_bytes, max_retries=3):
    """ يرسل طلبًا إلى واجهة برمجة تطبيقات LuminAI ويعيد الترجمة. """
    payload = { "content": prompt, "imageBuffer": list(image_bytes), "options": {"clean_output": True} }
    headers = { "Content-Type": "application/json", "Accept-Language": "ar" }

    for attempt in range(max_retries):
        try:
            response = requests.post(LUMINAI_API_URL, json=payload, headers=headers, timeout=35) # زيادة المهلة
            if response.status_code == 200:
                result_text = response.json().get("result", "")
                translation = extract_translation(result_text.strip())
                return translation
            elif response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 5))
                print(f"⚠️ Rate limit hit (429). Retrying after {retry_after} seconds...")
                time.sleep(retry_after)
            else:
                print(f"❌ LuminAI request failed: {response.status_code} {response.text}")
                return "" # فشل غير قابل للمحاولة
        except RequestException as e:
            print(f"❌ Network error during LuminAI request (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1: return ""
            time.sleep(2 * (attempt + 1))
        except Exception as e:
             print(f"❌ Unexpected error during LuminAI request (Attempt {attempt + 1}/{max_retries}): {e}")
             if attempt == max_retries - 1: return ""
             time.sleep(2)
    return ""

def find_optimal_text_settings_final(draw, text, initial_shrunk_polygon):
    """ يبحث عن أفضل حجم خط وتنسيق نص. (الكود الأصلي مع تعديلات طفيفة لاستخدام text_formatter) """
    if not initial_shrunk_polygon or initial_shrunk_polygon.is_empty or not initial_shrunk_polygon.is_valid:
        print("⚠️ Invalid polygon in find_optimal_text_settings.")
        return None
    if not text: return None

    best_fit = None
    for font_size in range(65, 4, -1):
        font = text_formatter.get_font(font_size)
        if font is None: continue

        padding_distance = max(1.5, font_size * 0.12)
        try:
            text_fitting_polygon = initial_shrunk_polygon.buffer(-padding_distance, join_style=2)
            if not text_fitting_polygon.is_valid or text_fitting_polygon.is_empty: text_fitting_polygon = initial_shrunk_polygon.buffer(-2.0, join_style=2)
            if not text_fitting_polygon.is_valid or text_fitting_polygon.is_empty or text_fitting_polygon.geom_type != 'Polygon': continue
        except Exception: continue # تجاهل أخطاء التصغير

        minx, miny, maxx, maxy = text_fitting_polygon.bounds
        target_width, target_height = maxx - minx, maxy - miny
        if target_width <= 5 or target_height <= 10: continue

        wrapped_text = text_formatter.layout_balanced_text(draw, text, font, target_width)
        if not wrapped_text: continue

        try:
            m_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=4, align='center')
            text_actual_width, text_actual_height = m_bbox[2] - m_bbox[0], m_bbox[3] - m_bbox[1]
            shadow_offset = max(1, font_size // 18)

            if (text_actual_height + shadow_offset) <= target_height and (text_actual_width + shadow_offset) <= target_width:
                x_offset, y_offset = (target_width - text_actual_width) / 2, (target_height - text_actual_height) / 2
                draw_x, draw_y = minx + x_offset - m_bbox[0], miny + y_offset - m_bbox[1]
                best_fit = {'text': wrapped_text, 'font': font, 'x': int(draw_x), 'y': int(draw_y), 'font_size': font_size}
                break # تم العثور على الأفضل
        except Exception as measure_err:
            print(f"⚠️ Error measuring text: {measure_err}")
            continue

    if best_fit is None: print("⚠️ Could not find suitable font size.")
    return best_fit

def draw_text_on_layer(text_settings, image_size):
    """ يرسم النص المحدد (مع الظل) على طبقة شفافة. (الكود الأصلي) """
    text_layer = Image.new('RGBA', image_size, (0, 0, 0, 0))
    draw_on_layer = ImageDraw.Draw(text_layer)
    font, text_to_draw, x, y, font_size = text_settings['font'], text_settings['text'], text_settings['x'], text_settings['y'], text_settings['font_size']
    shadow_offset = max(1, font_size // 18)
    shadow_color_with_alpha = SHADOW_COLOR + (SHADOW_OPACITY,)
    # Draw shadow
    draw_on_layer.multiline_text((x + shadow_offset, y + shadow_offset), text_to_draw, font=font, fill=shadow_color_with_alpha, align='center', spacing=4)
    # Draw text
    draw_on_layer.multiline_text((x, y), text_to_draw, font=font, fill=TEXT_COLOR + (255,), align='center', spacing=4)
    return text_layer

# --- الدوال الرئيسية للمعالجة ---

def remove_text_from_image(image):
    """يزيل النص من الصورة باستخدام Roboflow و inpainting."""
    print("Step 1: Removing original text...")
    start_time = time.time()
    result_image = image.copy()
    text_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # ترميز الصورة لـ Roboflow
    is_success, buffer = cv2.imencode('.jpg', image)
    if not is_success:
        print("❌ Error encoding image for Roboflow.")
        return result_image # إرجاع الصورة الأصلية عند الفشل
    b64_image = base64.b64encode(buffer).decode('utf-8')

    try:
        print(f"   Sending request to Roboflow text detection...")
        response_text = requests.post(
            f'https://serverless.roboflow.com/text-detection-w0hkg/1?api_key={ROBOFLOW_TEXT_API_KEY}',
            data=b64_image,
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            timeout=25
        )
        response_text.raise_for_status() # إطلاق استثناء لأخطاء HTTP
        data_text = response_text.json()
        text_predictions = data_text.get("predictions", [])
        print(f"   Found {len(text_predictions)} potential text areas.")

        polygons_drawn = 0
        for pred in text_predictions:
            points = pred.get("points", [])
            if len(points) >= 3:
                polygon_np = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)
                try:
                    cv2.fillPoly(text_mask, [polygon_np], 255)
                    polygons_drawn += 1
                except Exception as fill_err:
                    print(f"⚠️ Warning: Error drawing text polygon: {fill_err}")

        print(f"   Created text mask with {polygons_drawn} polygons.")

        if np.any(text_mask):
            print(f"   Inpainting detected text areas...")
            result_image = cv2.inpaint(image, text_mask, 10, cv2.INPAINT_NS)
            print(f"   Inpainting complete.")
        else:
            print(f"   No text detected, skipping inpainting.")

    except RequestException as req_err:
        print(f"❌ Network error during Roboflow text detection: {req_err}.")
    except Exception as e:
        print(f"❌ Error during text detection/inpainting: {e}.")

    print(f"Step 1 finished in {time.time() - start_time:.2f} seconds.")
    return result_image

def detect_bubbles(image):
    """يكتشف فقاعات الكلام باستخدام Roboflow."""
    print("\nStep 2: Detecting speech bubbles...")
    start_time = time.time()
    bubble_predictions = []

    is_success, buffer = cv2.imencode('.jpg', image)
    if not is_success:
        print("❌ Error encoding image for bubble detection.")
        return []
    b64_image = base64.b64encode(buffer).decode('utf-8')

    try:
        print(f"   Sending request to Roboflow bubble detection...")
        response_bubbles = requests.post(
            f'https://outline.roboflow.com/yolo-0kqkh/2?api_key={ROBOFLOW_BUBBLE_API_KEY}',
            data=b64_image,
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            timeout=25
        )
        response_bubbles.raise_for_status()
        data_bubbles = response_bubbles.json()
        bubble_predictions = data_bubbles.get("predictions", [])
        print(f"   Found {len(bubble_predictions)} speech bubbles.")

    except RequestException as req_err:
        print(f"❌ Network error during Roboflow bubble detection: {req_err}.")
    except Exception as e:
        print(f"❌ Error during Roboflow bubble detection: {e}.")

    print(f"Step 2 finished in {time.time() - start_time:.2f} seconds.")
    return bubble_predictions

def translate_and_draw(image, bubble_predictions, original_image_for_cropping):
    """يترجم النص ويرسمه داخل الفقاعات المكتشفة."""
    print("\nStep 3: Translating and drawing text in bubbles...")
    start_time = time.time()
    if not bubble_predictions:
        print("   No bubbles provided, skipping translation drawing.")
        return image # إرجاع الصورة كما هي

    try:
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert('RGBA')
    except Exception as pil_conv_err:
        print(f"❌ Error converting result image to PIL format: {pil_conv_err}")
        return image # إرجاع الصورة الأصلية عند الفشل

    image_size = image_pil.size
    # نحتاج لـ Draw مؤقت لقياس النص داخل find_optimal_text_settings_final
    temp_draw_for_settings = ImageDraw.Draw(Image.new('RGBA', image_size))

    bubble_count = len(bubble_predictions)
    processed_count = 0

    for i, pred in enumerate(bubble_predictions):
        print(f"\nProcessing bubble {i + 1}/{bubble_count}...")
        points = pred.get("points", [])
        if len(points) < 3:
            print("   Skipping bubble: Not enough points.")
            continue

        coords = [(int(p["x"]), int(p["y"])) for p in points]

        try:
            original_polygon = Polygon(coords)
            if not original_polygon.is_valid:
                original_polygon = make_valid(original_polygon)
                if original_polygon.geom_type == 'MultiPolygon':
                    original_polygon = max(original_polygon.geoms, key=lambda p: p.area, default=None)
                if not isinstance(original_polygon, Polygon) or original_polygon.is_empty or not original_polygon.is_valid:
                    print("   Skipping bubble: Polygon validation failed.")
                    continue

            # قص الصورة الأصلية للترجمة
            minx_orig, miny_orig, maxx_orig, maxy_orig = map(int, original_polygon.bounds)
            h_img, w_img = original_image_for_cropping.shape[:2]
            minx_orig, miny_orig = max(0, minx_orig), max(0, miny_orig)
            maxx_orig, maxy_orig = min(w_img, maxx_orig), min(h_img, maxy_orig)

            if maxx_orig <= minx_orig or maxy_orig <= miny_orig: continue
            bubble_crop = original_image_for_cropping[miny_orig:maxy_orig, minx_orig:maxx_orig]
            if bubble_crop.size == 0: continue

            _, crop_buffer = cv2.imencode('.jpg', bubble_crop)
            if crop_buffer is None: continue
            crop_bytes = crop_buffer.tobytes()

            # الحصول على الترجمة
            print("   Requesting translation from LuminAI...")
            translation_prompt = 'ترجم نص المانجا هذا إلى اللغة العربية بحيث تكون الترجمة مفهومة وتوصل المعنى الى القارئ. أرجو إرجاع الترجمة فقط بين علامتي اقتباس مثل "النص المترجم". مع مراعاة النبرة والانفعالات الظاهرة في كل سطر (مثل: الصراخ، التردد، الهمس) وأن تُترجم بطريقة تُحافظ على الإيقاع المناسب للفقاعة.'
            translation = ask_luminai(translation_prompt, crop_bytes)

            if not translation:
                print("   Skipping bubble: Failed to get translation.")
                continue
            print(f"   Translation received: '{translation}'")

            # تصغير المضلع مبدئياً لوضع النص
            width_orig, height_orig = maxx_orig - minx_orig, maxy_orig - miny_orig
            initial_buffer_distance = max(3.0, (width_orig + height_orig) / 2 * 0.10)
            try:
                 initial_shrunk_polygon = original_polygon.buffer(-initial_buffer_distance, join_style=2)
                 if not initial_shrunk_polygon.is_valid or initial_shrunk_polygon.is_empty: initial_shrunk_polygon = original_polygon.buffer(-3.0, join_style=2)
                 if not initial_shrunk_polygon.is_valid or initial_shrunk_polygon.is_empty or initial_shrunk_polygon.geom_type != 'Polygon':
                      print("   Warning: Could not create valid shrunk polygon. Using original boundary.")
                      initial_shrunk_polygon = original_polygon
            except Exception:
                 print(f"   Warning: Error shrinking polygon. Using original boundary.")
                 initial_shrunk_polygon = original_polygon

            if not isinstance(initial_shrunk_polygon, Polygon) or initial_shrunk_polygon.is_empty or not initial_shrunk_polygon.is_valid:
                print("   Skipping bubble: Final polygon is invalid.")
                continue

            # تنسيق النص العربي
            arabic_text = text_formatter.format_arabic_text(translation)
            if not arabic_text:
                print("   Skipping bubble: Formatted Arabic text is empty.")
                continue

            # البحث عن أفضل إعدادات للنص
            print("   Finding optimal font size and layout...")
            text_settings = find_optimal_text_settings_final(
                temp_draw_for_settings, # للقياسات
                arabic_text,
                initial_shrunk_polygon
            )

            # الرسم والدمج
            if text_settings:
                print(f"   Optimal settings found: Font Size {text_settings['font_size']}")
                print("   Drawing text onto layer...")
                text_layer = draw_text_on_layer(text_settings, image_size)
                print("   Compositing text layer...")
                image_pil.paste(text_layer, (0, 0), text_layer)
                processed_count += 1
            else:
                print("   Skipping bubble: Could not fit text.")

        except Exception as bubble_proc_err:
            print(f"❌ Error processing bubble {i + 1}: {bubble_proc_err}")
            import traceback
            traceback.print_exc() # طباعة تتبع الخطأ للمساعدة في التصحيح
            continue # الانتقال إلى الفقاعة التالية

    print(f"\nProcessed {processed_count}/{bubble_count} bubbles with translated text.")
    print(f"Step 3 finished in {time.time() - start_time:.2f} seconds.")

    # تحويل الصورة النهائية مرة أخرى إلى تنسيق OpenCV BGR
    try:
        final_image_rgb = image_pil.convert('RGB')
        final_image_np = cv2.cvtColor(np.array(final_image_rgb), cv2.COLOR_RGB2BGR)
        return final_image_np
    except Exception as final_conv_err:
        print(f"❌ Error converting final PIL image back to OpenCV: {final_conv_err}")
        return image # إرجاع الصورة قبل رسم آخر نص في حالة الفشل

