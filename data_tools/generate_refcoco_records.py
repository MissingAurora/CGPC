"""Generate ALL and SINGLE conversation records from existing COCO image splits."""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCES_PATH = os.path.join(DATA_DIR, "instances.json")
SRC_BASE = os.path.join(DATA_DIR, "train2014_7")
OUTPUT_DIR = SRC_BASE

# 80 个 COCO 类别 英文 → 中文 映射
CAT_CN = {
    "person": "人", "bicycle": "自行车", "car": "汽车", "motorcycle": "摩托车",
    "airplane": "飞机", "bus": "公交车", "train": "火车", "truck": "卡车",
    "boat": "船", "traffic light": "交通灯", "fire hydrant": "消防栓",
    "stop sign": "停车标志", "parking meter": "停车计时器", "bench": "长凳",
    "bird": "鸟", "cat": "猫", "dog": "狗", "horse": "马", "sheep": "羊",
    "cow": "牛", "elephant": "大象", "bear": "熊", "zebra": "斑马",
    "giraffe": "长颈鹿", "backpack": "背包", "umbrella": "雨伞",
    "handbag": "手提包", "tie": "领带", "suitcase": "行李箱",
    "frisbee": "飞盘", "skis": "滑雪板", "snowboard": "滑雪板",
    "sports ball": "运动球", "kite": "风筝", "baseball bat": "棒球棒",
    "baseball glove": "棒球手套", "skateboard": "滑板", "surfboard": "冲浪板",
    "tennis racket": "网球拍", "bottle": "瓶子", "wine glass": "酒杯",
    "cup": "杯子", "fork": "叉子", "knife": "刀", "spoon": "勺子",
    "bowl": "碗", "banana": "香蕉", "apple": "苹果", "sandwich": "三明治",
    "orange": "橙子", "broccoli": "西兰花", "carrot": "胡萝卜",
    "hot dog": "热狗", "pizza": "披萨", "donut": "甜甜圈", "cake": "蛋糕",
    "chair": "椅子", "couch": "沙发", "potted plant": "盆栽",
    "bed": "床", "dining table": "餐桌", "toilet": "马桶",
    "tv": "电视", "laptop": "笔记本电脑", "mouse": "鼠标",
    "remote": "遥控器", "keyboard": "键盘", "cell phone": "手机",
    "microwave": "微波炉", "oven": "烤箱", "toaster": "烤面包机",
    "sink": "水槽", "refrigerator": "冰箱", "book": "书",
    "clock": "时钟", "vase": "花瓶", "scissors": "剪刀",
    "teddy bear": "泰迪熊", "hair drier": "吹风机", "toothbrush": "牙刷",
}

HUMAN_TEMPLATE = (
    "<image>\n请定位图中所有{cn}，按照从左往右，"
    "从上到下的顺序依次输出每一个{cn}的bbox，每个{cn}一个，"
    "不允许合并或遗漏。按以下格式输出:"
    '{{"bbox_2d":[……],"labels":{en}}}'
)

# 单目标定位模板：随机要求最左侧/最右侧/最中间
SINGLE_HUMAN_TEMPLATE = (
    "<image>\n请定位图中{pos}的{cn}，输出相应{cn}的bbox，每个{cn}一个，"
    "不允许合并或遗漏。按以下格式输出:"
    '{{"bbox_2d":[……],"labels":{en}}}'
)

# 三个随机条件
POS_CN = ("最左侧", "最右侧", "最中间")


# 图片数量太少、不再标注的类别（按英文名，整条删除对应图片的标注）
EXCLUDED_CATS = {
    "bicycle", "sandwich", "sports ball", "potted plant", "kite",
    "knife", "bench", "remote", "traffic light", "toilet",
    "surfboard", "spoon", "fork", "cell phone", "horse",
    "truck", "airplane", "train", "tie", "clock",
    "toothbrush", "refrigerator", "bus", "laptop", "tv",
    "skis", "bed", "baseball bat", "backpack", "dog",
    "scissors", "keyboard", "frisbee", "giraffe", "cat",
    "parking meter", "dining table", "tennis racket", "skateboard", "sink",
    "oven", "mouse", "handbag", "snowboard","fire hydrant",
}


def load_data():
    with open(INSTANCES_PATH, 'r', encoding='utf-8-sig') as f:
        coco = json.load(f)

    # file_name stem -> 图片信息
    name_to_img = {}
    for img in coco['images']:
        stem = os.path.splitext(img['file_name'])[0]
        name_to_img[stem] = img

    # 建立索引
    images = {img['id']: img for img in coco['images']}
    cats = {c['id']: c['name'] for c in coco['categories']}

    # 每张图的标注按 category_id 分组
    # 同时收集所有 bbox
    img_anns = defaultdict(list)          # image_id -> [annotation]
    img_cat_count = defaultdict(Counter)  # image_id -> {category_id: count}

    for ann in coco['annotations']:
        img_anns[ann['image_id']].append(ann)
        img_cat_count[ann['image_id']][ann['category_id']] += 1

    return name_to_img, images, cats, img_anns, img_cat_count


def coco_bbox_to_xyxy(bbox):
    """COCO [x, y, w, h] → [x1, y1, x2, y2]"""
    x, y, w, h = bbox
    return [round(x, 1), round(y, 1), round(x + w, 1), round(y + h, 1)]


def normalize_bbox_to_1000(bbox, w, h):
    """将 [x1,y1,x2,y2] 按图片尺寸归一化到 0~1000（保留整数）"""
    x1, y1, x2, y2 = bbox
    xs = [round(x1 / w * 1000), round(x2 / w * 1000)]
    ys = [round(y1 / h * 1000), round(y2 / h * 1000)]
    return [max(0, min(1000, v)) for v in (xs[0], ys[0], xs[1], ys[1])]


def select_target(bboxes, mode, img_center_x):
    """从同类 bbox 中选出最左/最右/最中间的目标，返回单个 bbox [x1,y1,x2,y2]"""
    if mode == "left":
        # 最左侧：所有 bbox 的最左侧位置(x1)最小
        return min(bboxes, key=lambda b: (b[0], b[1]))
    if mode == "right":
        # 最右侧：所有 bbox 的最右侧位置(x2)最大
        return max(bboxes, key=lambda b: (b[2], b[3]))
    # 最中间：bbox 的中心横向坐标 (x1+x2)/2 与图片横向中心最接近
    return min(bboxes, key=lambda b: abs(((b[0] + b[2]) / 2.0) - img_center_x))


def process_split(split_name, name_to_img, images, cats,
                  img_anns, img_cat_count, output_name=None):
    """处理 train 或 val 分片，返回 JSON list"""
    src_dir = os.path.join(SRC_BASE, split_name)
    out_path = os.path.join(OUTPUT_DIR, f"refcoco_{output_name or split_name}.json")

    if not os.path.isdir(src_dir):
        raise FileNotFoundError(src_dir)

    files = sorted(f for f in os.listdir(src_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    print(f"\n[{split_name}] 共 {len(files)} 张图片")

    results = []
    skipped = 0
    excluded = 0
    pos_dist = Counter()  # 最左侧/最右侧/最中间 分布

    for fname in files:
        stem = os.path.splitext(fname)[0]
        img_info = name_to_img.get(stem)
        if img_info is None:
            skipped += 1
            continue

        img_id = img_info['id']
        w, h = img_info['width'], img_info['height']

        # 找出该图片数量最多的类别
        cat_counts = img_cat_count.get(img_id, Counter())
        if not cat_counts:
            skipped += 1
            continue

        # 取数量最多的类别（并列取 id 最小的）
        dom_cid = max(cat_counts.items(), key=lambda x: (x[1], -x[0]))[0]
        dom_cat_en = cats[dom_cid]
        dom_cat_cn = CAT_CN.get(dom_cat_en, dom_cat_en)

        # 图片数量太少的类别，整条删除对应图片的标注
        if dom_cat_en in EXCLUDED_CATS:
            excluded += 1
            continue

        # 收集该类别所有 bbox（按 imagesize 归一化到 0~1000）
        bboxes = [
            normalize_bbox_to_1000(coco_bbox_to_xyxy(ann['bbox']), w, h)
            for ann in img_anns.get(img_id, [])
            if ann['category_id'] == dom_cid
        ]

        if not bboxes:
            skipped += 1
            continue

        # 与提示词要求一致：按 从左往右(x1)、从上到下(y1) 排序输出
        bboxes.sort(key=lambda b: (b[0], b[1]))

        # 构建 human / gpt（所有目标）
        human_value = HUMAN_TEMPLATE.format(cn=dom_cat_cn, en=dom_cat_en)
        gpt_value = json.dumps({
            "bbox_2d": bboxes,
            "labels": [dom_cat_en] * len(bboxes),
        }, ensure_ascii=False)

        # 单目标定位：随机选 最左侧/最右侧/最中间
        pos = random.choice(POS_CN)
        pos_dist[pos] += 1
        mode = {"最左侧": "left", "最右侧": "right", "最中间": "middle"}[pos]
        # bbox 已按宽度归一化到 0~1000，图片横向中心即为 500
        target = select_target(bboxes, mode, img_center_x=500.0)
        single_human_value = SINGLE_HUMAN_TEMPLATE.format(pos=pos, cn=dom_cat_cn, en=dom_cat_en)
        single_gpt_value = json.dumps({
            "bbox_2d": [target],
            "labels": [dom_cat_en],
        }, ensure_ascii=False)

        # images 字段使用相对子目录路径（与 new_train.json 一致，方便直接定位图片）
        img_rel = f"{split_name}/{fname}"

        # 拆分：对话 1（定位所有同类目标）作为独立一条记录，conversations 为单层（无多余中括号）
        entry_all = {
            "conversations": [
                {"from": "human", "value": human_value},
                {"from": "gpt", "value": gpt_value},
            ],
            "images": [img_rel],
            "imagesize": [w, h],
        }
        # 拆分：对话 2（单目标定位 最左/最右/最中间）作为独立一条记录
        entry_single = {
            "conversations": [
                {"from": "human", "value": single_human_value},
                {"from": "gpt", "value": single_gpt_value},
            ],
            "images": [img_rel],
            "imagesize": [w, h],
        }
        results.append(entry_all)
        results.append(entry_single)

    # 写文件
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  ✅ 已保存: {out_path}")
    print(f"  📊 条目数: {len(results)} (每张图拆为2条: 全目标 + 单目标), "
          f"跳过: {skipped}, 排除(数量太少): {excluded}")
    print(f"  🎲 单目标条件分布: 最左侧 {pos_dist.get('最左侧', 0)}, "
          f"最右侧 {pos_dist.get('最右侧', 0)}, 最中间 {pos_dist.get('最中间', 0)}")

    # 统计各类别出现次数（conversations 现为单层，对话 1 为所有目标）
    cat_dist = Counter()
    for r in results:
        val = json.loads(r['conversations'][1]['value'])
        cat_dist[val['labels'][0]] += 1

    print(f"  📈 各类别数量:")
    for cname, cnt in cat_dist.most_common():
        cn_name = CAT_CN.get(cname, cname)
        print(f"    {cname:<16} ({cn_name}) : {cnt}")

    return results


def main():
    global INSTANCES_PATH, SRC_BASE, OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instances', required=True, help='COCO instance annotations')
    parser.add_argument('--split-dir', required=True, help='Existing train/ and valid/ image directories')
    parser.add_argument('--output-dir', required=True, help='Export directory')
    parser.add_argument('--valid-subdir', choices=['valid', 'val'], default='valid')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    INSTANCES_PATH, SRC_BASE, OUTPUT_DIR = args.instances, args.split_dir, args.output_dir
    for split in ('train', args.valid_subdir):
        if not os.path.isdir(os.path.join(SRC_BASE, split)):
            parser.error(f'Missing image directory: {os.path.join(SRC_BASE, split)}')
    for name in ('refcoco_train.json', 'refcoco_valid.json'):
        if os.path.exists(os.path.join(OUTPUT_DIR, name)):
            parser.error(f'Output already exists: {name}; choose a new output directory')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 固定随机种子，保证 train/val 的单目标条件（最左/最右/最中）分布可复现
    random.seed(args.seed)
    print("加载 instances.json ...")
    name_to_img, images, cats, img_anns, img_cat_count = load_data()
    print(f"  总图片数: {len(images)}, 总标注数: {sum(len(v) for v in img_anns.values())}")

    process_split("train", name_to_img, images, cats, img_anns, img_cat_count)
    process_split(args.valid_subdir, name_to_img, images, cats, img_anns,
                  img_cat_count, output_name='valid')

    print("\n✅ 全部完成！")


if __name__ == "__main__":
    main()
