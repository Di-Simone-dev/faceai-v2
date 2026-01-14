import os

def rename_images(folder):
    for i in range(1, 25):
        old_name = os.path.join(folder, f"face_{i}.jpg")
        new_name = os.path.join(folder, f"{i:06d}.png")

        if os.path.exists(old_name):
            os.rename(old_name, new_name)
            print(f"{old_name} -> {new_name}")
        else:
            print(f"File non trovato: {old_name}")

# Usa il percorso della cartella che contiene le immagini
rename_images("images_256")