import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from utils import show_masks,show_points,show_box,show_mask,show_mask_v,get_bounding_box
from sam2.build_sam import build_sam2_video_predictor
import yaml
import argparse


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='创建文件夹和子文件夹')
    parser.add_argument('folder_name', type=str, help='主文件夹的名称')
    args = parser.parse_args()


    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")
    #使用bfloat16精度
    #如果是Ampere架构（版本号>8）则启用tensorfloat-32精度，提升矩阵乘法和卷积运算的性能
    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    np.random.seed(3) #每次都是相同随机数序列，结果可重复

    # from sam2.build_sam import build_sam2
    sam2_checkpoint="../../sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml"
    class_name=args.folder_name
    # sam2_model=build_sam2(model_cfg,sam2_checkpoint,device=device)

    predictor2=build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=0)
    video_dir = os.path.join('../../m_data', 'raw_img',class_name)
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG",".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))



    inference_state = predictor2.init_state(video_path=video_dir)
    points = np.array([[160, 160],[160, 250]], dtype=np.float32)
    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    # Let's add a positive click at (x, y) = (210, 350) to get started
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1,1], np.int32)
    _, out_obj_ids, out_mask_logits = predictor2.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
        # box=box,
    )


    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor2.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    yaml_file_path='../../m_data/data.yaml'
    if not os.path.exists(yaml_file_path):
        data = {
            # 添加训练、验证和测试文件夹
            'train':'../../m_data/train',
            'val':'../../m_data/valid',
            'test':'../../m_data/test',
            'nc':0,
            'names': []
            }
        with open('../../m_data/data.yaml', 'w') as file:
            yaml.dump(data, file)

    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)
        # 添加类别数量和名称
        items = os.listdir('../../m_data/raw_img')
        data['nc'] = sum(1 for item in items if os.path.isdir(os.path.join('../../m_data/raw_img', item)))

        data['names']=[item for item in os.listdir('../../m_data/raw_img') if os.path.isdir(os.path.join('../../m_data/raw_img', item))]

    # 写回 data.yaml 文件
    with open('../../m_data/data.yaml', 'w') as file:
        yaml.dump(data, file)

    print("data.yaml 文件已更新。")
    # 获取文件夹中的所有项
    items = os.listdir('../../m_data/raw_img')
    # 统计子文件夹的数量
    subfolder_count = sum(1 for item in items if os.path.isdir(os.path.join('../../m_data/raw_img', item)))
    # class_id=subfolder_count-1
    class_id=int(args.folder_name[0])

    # render the segmentation results every few frames
    vis_frame_stride = 3
    plt.close("all")
    #train
    for out_frame_idx in range(0, int(7*len(frame_names)/10), 1):
        stat='train'
        YOYO_img_dir = os.path.join('../../m_data',stat, 'images')
        YOYO_lab_dir = os.path.join('../../m_data',stat, 'labels')
        os.makedirs(YOYO_img_dir, exist_ok=True)
        os.makedirs(YOYO_lab_dir, exist_ok=True)  
        image=Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
        # plt.figure(figsize=(6, 4))
        # plt.title(f"frame {out_frame_idx}")
        # plt.imshow(image)
        output_image_path = os.path.join(YOYO_img_dir, str(class_name)+'_'+frame_names[out_frame_idx])
        image.save(output_image_path)
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            # print(out_mask.shape)
            x0,y0,x1,y1=get_bounding_box(out_mask[0])
            # show_mask_v(out_mask, plt.gca(), obj_id=out_obj_id)#叠加binary图像
            # print("yolo:data:" ,(x0+x1)/2/out_mask.shape[1], (y0+y1)/2/out_mask.shape[2],(x1-x0)/out_mask.shape[1],(y1-y0)/out_mask.shape[2])
            values=[class_id,(x0+x1)/2/out_mask.shape[1], (y0+y1)/2/out_mask.shape[2],(x1-x0)/out_mask.shape[1],(y1-y0)/out_mask.shape[2]]
            formatted_values = ' '.join(str(value) for value in values)
            output_txt_path = os.path.join(YOYO_lab_dir, str(class_name)+'_'+frame_names[out_frame_idx].rstrip('.jpg')+'.txt')
            with open(output_txt_path, 'w') as f:
                f.write(formatted_values)
            plt.gca().add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, edgecolor='green', facecolor=(0, 0, 0, 0),lw=2))
            #创建一个txt文件存储

    for out_frame_idx in range(int(7*len(frame_names)/10), int(9*len(frame_names)/10), 1):
        stat='valid'
        YOYO_img_dir = os.path.join('../../m_data',stat, 'images')
        YOYO_lab_dir = os.path.join('../../m_data',stat, 'labels')
        os.makedirs(YOYO_img_dir, exist_ok=True)
        os.makedirs(YOYO_lab_dir, exist_ok=True)  
        image=Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
        # plt.figure(figsize=(6, 4))
        # plt.title(f"frame {out_frame_idx}")
        # plt.imshow(image)
        output_image_path = os.path.join(YOYO_img_dir, str(class_name)+'_'+frame_names[out_frame_idx])
        image.save(output_image_path)
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            # print(out_mask.shape)
            x0,y0,x1,y1=get_bounding_box(out_mask[0])
            # show_mask_v(out_mask, plt.gca(), obj_id=out_obj_id)#叠加binary图像
            # print("yolo:data:" ,(x0+x1)/2/out_mask.shape[1], (y0+y1)/2/out_mask.shape[2],(x1-x0)/out_mask.shape[1],(y1-y0)/out_mask.shape[2])
            values=[class_id,(x0+x1)/2/out_mask.shape[1], (y0+y1)/2/out_mask.shape[2],(x1-x0)/out_mask.shape[1],(y1-y0)/out_mask.shape[2]]
            formatted_values = ' '.join(str(value) for value in values)
            output_txt_path = os.path.join(YOYO_lab_dir, str(class_name)+'_'+frame_names[out_frame_idx].rstrip('.jpg')+'.txt')
            with open(output_txt_path, 'w') as f:
                f.write(formatted_values)
            plt.gca().add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, edgecolor='green', facecolor=(0, 0, 0, 0),lw=2))

    for out_frame_idx in range(int(9*len(frame_names)/10), len(frame_names), 1):
        stat='test'
        YOYO_img_dir = os.path.join('../../m_data',stat, 'images')
        YOYO_lab_dir = os.path.join('../../m_data',stat, 'labels')
        os.makedirs(YOYO_img_dir, exist_ok=True)
        os.makedirs(YOYO_lab_dir, exist_ok=True)  
        image=Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
        # plt.figure(figsize=(6, 4))
        # plt.title(f"frame {out_frame_idx}")
        # plt.imshow(image)
        output_image_path = os.path.join(YOYO_img_dir, str(class_name)+'_'+frame_names[out_frame_idx])
        image.save(output_image_path)
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            # print(out_mask.shape)
            x0,y0,x1,y1=get_bounding_box(out_mask[0])
            # show_mask_v(out_mask, plt.gca(), obj_id=out_obj_id)#叠加binary图像
            # print("yolo:data:" ,(x0+x1)/2/out_mask.shape[1], (y0+y1)/2/out_mask.shape[2],(x1-x0)/out_mask.shape[1],(y1-y0)/out_mask.shape[2])
            values=[class_id,(x0+x1)/2/out_mask.shape[1], (y0+y1)/2/out_mask.shape[2],(x1-x0)/out_mask.shape[1],(y1-y0)/out_mask.shape[2]]
            formatted_values = ' '.join(str(value) for value in values)
            output_txt_path = os.path.join(YOYO_lab_dir, str(class_name)+'_'+frame_names[out_frame_idx].rstrip('.jpg')+'.txt')
            with open(output_txt_path, 'w') as f:
                f.write(formatted_values)
            plt.gca().add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, edgecolor='green', facecolor=(0, 0, 0, 0),lw=2))