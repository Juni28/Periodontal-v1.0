import os

label_dir = "Dataset/Training/Labels"
classes_found = set()

for file in os.listdir(label_dir):
    if file.endswith(".txt"):
        with open(os.path.join(label_dir, file)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # Ignorar líneas vacías
                parts = line.split()
                if len(parts) < 1:
                    continue  # Ignorar líneas mal formateadas
                class_id = int(parts[0])
                classes_found.add(class_id)

print("Clases encontradas:", sorted(classes_found))
print("Número total de clases:", len(classes_found))

