import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from flask_cors import CORS
import hashlib
import io
import json
import os
import time

app = Flask(__name__)
CORS(app)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()


def load_tang_poetry_from_current_dir():
    poetry_dataset = []
    poetry_categories = {}
    category_keywords = {
        "山水类": ["山", "水", "峰", "瀑", "川", "湖", "海", "江", "河", "松", "石"],
        "田园类": ["田", "园", "农", "桑", "麻", "村", "舍", "耕", "种", "禾", "麦"],
        "雪景类": ["雪", "霜", "冰", "寒", "白", "冻", "凌"],
        "四季类": ["春", "夏", "秋", "冬", "晴", "雨", "风", "露"],
        "送别类": ["送", "别", "归", "行", "旅", "离", "饯"],
        "思乡类": ["乡", "亲", "家", "故", "归", "远"]
    }

    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"当前脚本目录：{current_dir}")

    single_tang_file = os.path.join(current_dir, "tang_poetry.json")
    if os.path.exists(single_tang_file):
        print(f"找到单个全唐诗文件：{single_tang_file}")
        try:
            with open(single_tang_file, "r", encoding="utf-8") as f:
                poem_list = json.load(f)

            poem_idx = 0
            for poem in poem_list:
                if isinstance(poem, dict):
                    poem_title = poem.get("title", "")
                    poem_content = poem.get("content", "")
                    if not poem_content and "paragraphs" in poem:
                        poem_content = "\n".join(p.strip() for p in poem["paragraphs"] if p.strip())
                elif isinstance(poem, list) and len(poem) >= 2:
                    poem_title = poem[0]
                    poem_content = poem[1]
                else:
                    continue

                if not poem_title or not poem_content:
                    continue
                if not poem_title.startswith("《"):
                    poem_title = f"《{poem_title}》"
                poem_content = "\n".join(line.strip() for line in poem_content.split("\n") if line.strip())

                poetry_dataset.append([poem_title, poem_content])
                # 自动分类
                poem_text = poem_title + poem_content
                category = "未分类"
                for cat_name, keywords in category_keywords.items():
                    if any(keyword in poem_text for keyword in keywords):
                        category = cat_name
                        break
                poetry_categories[poem_idx] = category
                poem_idx += 1

            print(f"从单个文件加载完成！共 {len(poetry_dataset)} 首诗歌")
            return poetry_dataset, poetry_categories
        except Exception as e:
            print(f"读取单个全唐诗文件失败：{e}，尝试读取拆分文件...")

    tang_file_prefix = "poet.tang."
    tang_file_suffix = ".json"
    tang_poetry_files = [
        os.path.join(current_dir, filename)
        for filename in os.listdir(current_dir)
        if filename.startswith(tang_file_prefix) and filename.endswith(tang_file_suffix)
    ]

    if len(tang_poetry_files) == 0:
        raise FileNotFoundError(
            f"在当前目录 {current_dir} 下未找到全唐诗文件！\n"
            f"请放置单个文件（如 tang_poetry.json）或拆分文件（如 poet.tang.0.json）到脚本同目录"
        )

    print(f"找到 {len(tang_poetry_files)} 个全唐诗拆分文件，开始加载...")
    poem_idx = 0
    for file_path in tang_poetry_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                poem_list = json.load(f)

            for poem in poem_list:
                poem_title = poem.get("title", "")
                if not poem_title:
                    continue
                if not poem_title.startswith("《"):
                    poem_title = f"《{poem_title}》"

                poem_paragraphs = poem.get("paragraphs", [])
                if not poem_paragraphs:
                    continue
                poem_content = "\n".join(p.strip() for p in poem_paragraphs if p.strip())
                if not poem_content:
                    continue

                poetry_dataset.append([poem_title, poem_content])
                # 自动分类
                poem_text = poem_title + poem_content
                category = "未分类"
                for cat_name, keywords in category_keywords.items():
                    if any(keyword in poem_text for keyword in keywords):
                        category = cat_name
                        break
                poetry_categories[poem_idx] = category
                poem_idx += 1

        except Exception as e:
            print(f"读取文件 {os.path.basename(file_path)} 失败，跳过：{e}")
            continue

    print(f"从拆分文件加载完成！共 {len(poetry_dataset)} 首诗歌")
    return poetry_dataset, poetry_categories

poetry_dataset, poetry_categories = load_tang_poetry_from_current_dir()
poetry_texts = [poem[1] for poem in poetry_dataset]

def batch_extract_text_features(text_list, batch_size=16):
    all_embeds = []
    total_batches = (len(text_list) + batch_size - 1) // batch_size
    print(f"开始提取文本特征，共 {len(text_list)} 条文本，分 {total_batches} 批次，批次大小：{batch_size}")

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=device == "cuda"):
            for i in range(0, len(text_list), batch_size):
                batch_start = time.time()
                batch_texts = text_list[i:i+batch_size]
                batch_idx = i // batch_size + 1


                batch_inputs = processor(
                    text=batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                ).to(device)


                batch_embeds = model.get_text_features(**batch_inputs).cpu().numpy()
                all_embeds.append(batch_embeds)


                if device == "cuda":
                    torch.cuda.empty_cache()


                batch_end = time.time()
                elapsed_time = batch_end - batch_start
                remaining_batches = total_batches - batch_idx
                estimated_remaining = elapsed_time * remaining_batches
                print(f"批次 {batch_idx}/{total_batches} 完成 | 单批次耗时：{elapsed_time:.2f}s | 预计剩余：{estimated_remaining:.2f}s")


    all_embeds_np = np.concatenate(all_embeds, axis=0)
    print(f"文本特征提取完成！特征形状：{all_embeds_np.shape}")
    return all_embeds_np


poetry_text_embeds = batch_extract_text_features(poetry_texts, batch_size=16)  # 可根据显存调整为8/32


feature_cache = {}
def get_data_md5(data):
    if isinstance(data, bytes):
        return hashlib.md5(data).hexdigest()
    else:
        return hashlib.md5(str(data).encode('utf-8')).hexdigest()

def extract_image_feature(image_file, use_cache=True):
    image_bytes = image_file.read()
    image_file.seek(0)
    cache_key = f"image_{get_data_md5(image_bytes)}"

    if use_cache and cache_key in feature_cache:
        return feature_cache[cache_key]

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_embed = model.get_image_features(**image_inputs).cpu().numpy()

    if use_cache:
        feature_cache[cache_key] = image_embed
    return image_embed

def extract_text_feature(text, use_cache=True):
    cache_key = f"text_{get_data_md5(text)}"

    if use_cache and cache_key in feature_cache:
        return feature_cache[cache_key]

    text_inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
    with torch.no_grad():
        text_embed = model.get_text_features(**text_inputs).cpu().numpy()

    if use_cache:
        feature_cache[cache_key] = text_embed
    return text_embed

def retrieve_poetry_core(embed, top_k=3, category_filter=None):
    similarity_scores = cosine_similarity(embed, poetry_text_embeds)[0]
    if category_filter is not None and category_filter in list(set(poetry_categories.values())):
        valid_indices = [idx for idx, cat in poetry_categories.items() if cat == category_filter]
        for idx in range(len(similarity_scores)):
            if idx not in valid_indices:
                similarity_scores[idx] = -1

    sorted_indices = [idx for idx in np.argsort(similarity_scores)[::-1] if similarity_scores[idx] != -1]
    top_k_poetry = []
    for idx in sorted_indices[:top_k]:
        poetry_info = {
            "title": poetry_dataset[idx][0],
            "content": poetry_dataset[idx][1],
            "similarity": round(float(similarity_scores[idx]), 4),
            "category": poetry_categories.get(idx, "未分类")
        }
        top_k_poetry.append(poetry_info)
    return top_k_poetry

def search_poetry_by_keyword(keyword, top_k=3):
    if not keyword:
        return []
    match_scores = []
    for idx, (title, content) in enumerate(poetry_dataset):
        match_count = title.count(keyword) + content.count(keyword)
        if match_count > 0:
            match_scores.append((idx, match_count))
    match_scores.sort(key=lambda x: x[1], reverse=True)

    top_k_poetry = []
    for idx, score in match_scores[:top_k]:
        poetry_info = {
            "title": poetry_dataset[idx][0],
            "content": poetry_dataset[idx][1],
            "match_count": score,
            "category": poetry_categories.get(idx, "未分类")
        }
        top_k_poetry.append(poetry_info)
    return top_k_poetry

def recommend_similar_poetry(title, top_k=3):
    target_idx = -1
    for idx, (poem_title, _) in enumerate(poetry_dataset):
        if poem_title == title:
            target_idx = idx
            break
    if target_idx == -1:
        return {"error": "未找到该诗歌"}

    target_category = poetry_categories.get(target_idx, "未分类")
    target_embed = poetry_text_embeds[target_idx:target_idx+1]
    similarity_scores = cosine_similarity(target_embed, poetry_text_embeds)[0]
    similarity_scores[target_idx] = -1

    valid_indices = [idx for idx, cat in poetry_categories.items() if cat == target_category and similarity_scores[idx] != -1]
    valid_indices.sort(key=lambda x: similarity_scores[x], reverse=True)

    similar_poetry = []
    for idx in valid_indices[:top_k]:
        poetry_info = {
            "title": poetry_dataset[idx][0],
            "content": poetry_dataset[idx][1],
            "similarity": round(float(similarity_scores[idx]), 4),
            "category": target_category
        }
        similar_poetry.append(poetry_info)
    return {"target_poetry": poetry_dataset[target_idx], "similar_poetry": similar_poetry}

def batch_process_images(image_files, top_k=3, category_filter=None):
    batch_results = []
    for img_file in image_files:
        try:
            image_embed = extract_image_feature(img_file)
            poetry_result = retrieve_poetry_core(image_embed, top_k, category_filter)
            batch_results.append({
                "filename": img_file.filename,
                "status": "success",
                "poetry": poetry_result
            })
        except Exception as e:
            batch_results.append({
                "filename": img_file.filename,
                "status": "failed",
                "error": str(e)
            })
    return batch_results

#
@app.route('/retrieve_poetry', methods=['POST'])
def retrieve_poetry_api():
    try:
        if 'image' not in request.files:
            return jsonify({"status": "failed", "error": "未上传图片"}), 400
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"status": "failed", "error": "图片文件名不能为空"}), 400

        top_k = request.form.get('top_k', 3, type=int)
        category_filter = request.form.get('category', None)
        image_embed = extract_image_feature(image_file)
        top_poetry = retrieve_poetry_core(image_embed, top_k, category_filter)

        return jsonify({"status": "success", "poetry": top_poetry})
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@app.route('/retrieve_poetry_by_text', methods=['POST'])
def retrieve_poetry_by_text_api():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"status": "failed", "error": "未提供文本描述"}), 400
        text = data['text']
        top_k = data.get('top_k', 3)
        category_filter = data.get('category', None)

        text_embed = extract_text_feature(text)
        top_poetry = retrieve_poetry_core(text_embed, top_k, category_filter)

        return jsonify({"status": "success", "poetry": top_poetry})
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@app.route('/search_poetry', methods=['GET', 'POST'])
def search_poetry_api():
    try:
        if request.method == 'GET':
            keyword = request.args.get('keyword', '')
            top_k = request.args.get('top_k', 3, type=int)
        else:
            data = request.get_json()
            keyword = data.get('keyword', '')
            top_k = data.get('top_k', 3, type=int)

        search_result = search_poetry_by_keyword(keyword, top_k)
        return jsonify({"status": "success", "count": len(search_result), "poetry": search_result})
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@app.route('/recommend_poetry', methods=['GET', 'POST'])
def recommend_poetry_api():
    try:
        if request.method == 'GET':
            title = request.args.get('title', '')
            top_k = request.args.get('top_k', 3, type=int)
        else:
            data = request.get_json()
            title = data.get('title', '')
            top_k = data.get('top_k', 3, type=int)

        recommend_result = recommend_similar_poetry(title, top_k)
        return jsonify({"status": "success", "data": recommend_result})
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@app.route('/batch_retrieve_poetry', methods=['POST'])
def batch_retrieve_poetry_api():
    try:
        if 'images' not in request.files:
            return jsonify({"status": "failed", "error": "未上传批量图片"}), 400
        image_files = request.files.getlist('images')
        if not image_files:
            return jsonify({"status": "failed", "error": "批量图片列表为空"}), 400

        top_k = request.form.get('top_k', 3, type=int)
        category_filter = request.form.get('category', None)
        batch_results = batch_process_images(image_files, top_k, category_filter)

        return jsonify({
            "status": "success",
            "total_count": len(image_files),
            "success_count": len([r for r in batch_results if r['status'] == 'success']),
            "results": batch_results
        })
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@app.route('/get_poetry_categories', methods=['GET'])
def get_poetry_categories_api():
    try:
        categories = list(set(poetry_categories.values()))
        return jsonify({"status": "success", "categories": categories})
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)