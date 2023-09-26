import os
import re
import matplotlib.pyplot as plt 


def show_data_distribution():
    # Set the path to the dataset folder
    dataset_folder = './dataset/'

    # Get the path to the labels folder
    labels_folder = os.path.join(dataset_folder, 'labels')

    # Get a list of all label files in the labels folder
    label_files = os.listdir(labels_folder)

    # Count the occurrences of each label
    label_counts = {}
    for label_file in label_files:
        with open(os.path.join(labels_folder, label_file), 'r') as file:
            label = file.read().strip()

        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1

    # Sort labels alphabetically
    sorted_labels = sorted(label_counts.keys(), key=lambda x: int(re.findall(r'\d+', x)[0]))


    # Create a bar chart to visualize the sorted label distribution
    counts = [label_counts[label] for label in sorted_labels]
    print(f'Number of images: {sum(counts)}')

    plt.bar(sorted_labels, counts)
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title('Label Distribution')
    plt.xticks(rotation=45)
    plt.show()
    return

def visualize_featureMaps(model, outputs, labels, num_images_to_show):
    feature_map = model.feature_map
    fig, axs = plt.subplots(1, num_images_to_show, figsize=(15, 3))
    for i in range(num_images_to_show):
        plt.subplot(1,num_images_to_show,i+1)
        plt.imshow(feature_map[i, 0].cpu().detach(), cmap='jet')
        plt.title(f"Count: {outputs[i][0]}")
        plt.axis('off')
        
    return
