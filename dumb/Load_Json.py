import json

def get_top_k_labels(json_path, k=300):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Count instances for each gloss
        gloss_instance_counts = []
        for item in data:
            gloss = item.get("gloss")
            instances = item.get("instances", [])
            if gloss and isinstance(instances, list):
                gloss_instance_counts.append((gloss, len(instances)))

        # Sort by number of instances, descending
        gloss_instance_counts.sort(key=lambda x: x[1], reverse=True)

        # Take top-k
        top_k_glosses = [gloss for gloss, _ in gloss_instance_counts[:k]]

        # Create mappings
        class_labels = top_k_glosses
        label_to_idx = {label: idx for idx, label in enumerate(class_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}

        print(f"✅ Extracted top {k} class labels.")
        return class_labels, label_to_idx, idx_to_label

    except Exception as e:
        print(f"❌ Error: {e}")
        return [], {}, {}




LABEL_JSON_PATH = r'C:\Users\Harsha PC\Desktop\deaf\WLASL_v0.3.json'
class_labels, label_to_idx, idx_to_label = get_top_k_labels(LABEL_JSON_PATH)
print(len(class_labels))
