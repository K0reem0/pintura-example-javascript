# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
from flask_cors import CORS
import manga_translator  # استيراد الوحدة التي أنشأناها
import base64
import traceback # للمساعدة في تتبع الأخطاء

app = Flask(__name__)
CORS(app)  # تفعيل CORS للسماح بالطلبات من الواجهة الأمامية

@app.route('/')
def home():
    return "Manga Translator Backend is running!"

@app.route('/whiten', methods=['POST'])
def whiten_endpoint():
    """نقطة نهاية لتبييض النص فقط."""
    print("\nReceived request for /whiten")
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "Missing 'image' in request body"}), 400

        base64_input = data['image']
        # إزالة بادئة البيانات إذا كانت موجودة (مثل 'data:image/jpeg;base64,')
        if ',' in base64_input:
             base64_input = base64_input.split(',', 1)[1]

        original_image = manga_translator.decode_image(base64_input)
        if original_image is None:
            return jsonify({"error": "Could not decode input image"}), 400

        whitened_image = manga_translator.remove_text_from_image(original_image)

        base64_output = manga_translator.encode_image(whitened_image)
        if base64_output is None:
            return jsonify({"error": "Could not encode processed image"}), 500

        print("Whiten process completed successfully.")
        return jsonify({"image": base64_output})

    except Exception as e:
        print(f"❌ Error in /whiten endpoint: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred during whitening"}), 500

@app.route('/translate_auto', methods=['POST'])
def translate_auto_endpoint():
    """نقطة نهاية للتبييض والترجمة والرسم التلقائي."""
    print("\nReceived request for /translate_auto")
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "Missing 'image' in request body"}), 400

        base64_input = data['image']
        if ',' in base64_input:
             base64_input = base64_input.split(',', 1)[1]

        original_image = manga_translator.decode_image(base64_input)
        if original_image is None:
            return jsonify({"error": "Could not decode input image"}), 400

        # الخطوة 1: التبييض
        whitened_image = manga_translator.remove_text_from_image(original_image)

        # الخطوة 2: اكتشاف الفقاعات (على الصورة المبيضة أو الأصلية؟ الأصلية أفضل عادةً للاكتشاف)
        # سنستخدم الصورة الأصلية لاكتشاف الفقاعات ولكن سنمرر الصورة المبيضة للترجمة والرسم
        bubble_predictions = manga_translator.detect_bubbles(original_image) # الاكتشاف على الأصلية

        if not bubble_predictions:
            print("No bubbles detected, returning whitened image.")
            base64_output = manga_translator.encode_image(whitened_image)
            if base64_output:
                 return jsonify({"image": base64_output})
            else:
                 return jsonify({"error": "Could not encode whitened image (no bubbles)"}), 500

        # الخطوة 3: الترجمة والرسم (على الصورة المبيضة)
        # نحتاج الصورة الأصلية لتمريرها لقص الفقاعات للترجمة
        final_image = manga_translator.translate_and_draw(
            whitened_image, # الرسم على الصورة المبيضة
            bubble_predictions,
            original_image # المرور بالأصلية لقص الترجمة
        )

        base64_output = manga_translator.encode_image(final_image)
        if base64_output is None:
            return jsonify({"error": "Could not encode final processed image"}), 500

        print("Translate Auto process completed successfully.")
        return jsonify({"image": base64_output})

    except Exception as e:
        print(f"❌ Error in /translate_auto endpoint: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred during auto translation"}), 500

if __name__ == '__main__':
    # الاستماع على جميع الواجهات المتاحة (مفيد لـ Docker أو الوصول الشبكي)
    # استخدم 127.0.0.1 بدلاً من 0.0.0.0 إذا كنت تريد الوصول المحلي فقط
    # debug=True يساعد في التطوير لكن يجب تعطيله في الإنتاج
    app.run(host='0.0.0.0', port=5000, debug=True)
