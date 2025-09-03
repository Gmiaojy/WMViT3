import matplotlib.pyplot as plt
import os
from PIL import Image
from utils.dataset import _fuse_images 

def visualize_fusion_methods(img_list, fusion_methods, output_dir=None):
    """
    Visualize and compare the effects of different image fusion methods.

    Args:
        img_list (list): A list containing the base paths of multiple samples.
        fusion_methods (list): A list of strings specifying the fusion method names to be displayed.
        output_dir (str, optional): If provided, the generated comparison image will be saved to this directory.
    """
    num_samples = len(img_list)
    num_fusion_methods = len(fusion_methods)
    num_original_images = 3 # S0, L1, L2

    # ceate a large figure to hold all subplots
    fig, axes = plt.subplots(
        num_samples,                                 # rows
        num_original_images + num_fusion_methods,    # cols
        figsize=(2.5 * (num_original_images + num_fusion_methods), 2.5 * num_samples),
        squeeze=False
    )

    # loop through each sample and fusion method to populate the subplots
    for i, base_path in enumerate(img_list):
        img_name = os.path.basename(base_path)
        cls_dir = os.path.dirname(base_path)
        category_name = os.path.basename(cls_dir)   # Extract category name for the row title
        base_name = img_name.rsplit('_', 1)[0]
        
        print(f"Processing sample {i+1}/{num_samples}: {base_name}")

        try:
            img_paths = [os.path.join(cls_dir, f"{base_name}_{j}.jpg") for j in range(num_original_images)]
            original_images_pil = [Image.open(p).convert('L') for p in img_paths]
        except FileNotFoundError:
            print(f"Warning: Could not find image files for sample {base_path}, this row will be left blank.")
            for ax in axes[i]:
                ax.axis('off')
            continue
        
        fused_images_pil = [ _fuse_images(original_images_pil, method=method) for method in fusion_methods]
        
        # Set row title
        axes[i,0].set_ylabel(
            category_name, 
            rotation=0, 
            size=18, 
            ha='right',
            va='center' 
        )
        # Display original images
        original_titles = ["I", "L1", "L2"]
        for j in range(num_original_images):
            axes[i, j].imshow(original_images_pil[j], cmap='gray')
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(original_titles[j], fontsize=22, pad=20)
            
        # Display fused images
        for j, method in enumerate(fusion_methods):
            col_idx = num_original_images + j
            axes[i, col_idx].imshow(fused_images_pil[j])
            axes[i, col_idx].axis('off')
            if i == 0:
                axes[i, col_idx].set_title(method, fontsize=22, pad=30)
                
            if category_name == 'ABS500' and method == 'rgb_0_1_2' and output_dir:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                # Build the save path and save the example picture
                save_path = os.path.join(output_dir, f"{category_name}_{method}.png")
                fused_images_pil[j].save(save_path)
                print(f"The specific picture has been saved to: {save_path}")
    
    # set total title   
    fig.suptitle(
        "Comparison of Different Image Fusion Methods",
        fontsize=24,
        fontweight='bold',
        ha='center'
    )
    fig.subplots_adjust(
        left=0.1,       # left margin
        right=0.98,     # right margin
        top=0.90,       # top margin
        bottom=0.02,    # bottom margin
        wspace=0.01,     # Horizontal spacing between subgraphs
        hspace=0.01    # Vertical spacing between sub-graphs
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "visualize_fusion_methods.png")
    try:
        plt.savefig(output_path, dpi=200, bbox_inches='tight') # bbox_inches='tight' to reduce whitespace
        print(f"\nFinal comparison image has been saved to: {output_path}")
    except Exception as e:
        print(f"Error occur in saving the file: {e}")
    finally:
        plt.close(fig) 



if __name__ == "__main__":
    img_list = [
        # Note: must download the full datas first
        'wmvit/datas/Input/ABS500/20250611_155745_126_1_0.jpg',
        'wmvit/datas/Input/PA6500/20250611_173053_827_1_0.jpg',
        'wmvit/datas/Input/PE500/20250612_103732_720_3_0.jpg',
        'wmvit/datas/Input/PP500/20250612_110453_129_1_0.jpg',
        'wmvit/datas/Input/PS500/20250611_174635_796_3_0.jpg'
    ]
    
    fusion_methods_to_test = [
        'rgb_0_1_2','rgb_1_0_2','rgb_1_2_0',
        'hls_0_1_2','hls_1_0_2','hls_1_2_0',
        'ycbcr_0_1_2'
    ]
    output_directory = 'wmvit/output/plot_fusion_methods'
    visualize_fusion_methods(img_list, fusion_methods_to_test, output_dir=output_directory)