import argparse
import os
import sys
from pathlib import Path
from turtle import clear

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


import tkinter as tk
import tkinter.ttk
from tkinter import *
import multiprocessing
import time
from PIL import Image, ImageTk, ImageDraw



# ==============================================================================
# Global Values
draw_points_list_all = []
draw_points_list = []
b1_moving = False
people_group = []
people_max = 0
# ==============================================================================




@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        data_dict=[], # the shared listed between the main and subprocess
):

    # print(f'{data_dict = }')
    print('data_dict = ', data_dict)

    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0-255 to 0.0-1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  
            seen += 1
            if webcam:  
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  
            s += '%gx%g ' % im.shape[2:]  

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            global people_group
            global people_max
            people_group = []

            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, " 

                # ____________________________________________________________________________________________________________
                # Write results
                # Group processing
                for *xyxy, conf, cls in reversed(det):
                    if view_img:  
                        c = int(cls)
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        
                        # ___________________________________________________                         
                        left_top_x = xyxy[0].data.cpu().numpy().tolist()
                        left_top_y = xyxy[1].data.cpu().numpy().tolist()
                        right_down_x = xyxy[2].data.cpu().numpy().tolist()
                        right_down_y = xyxy[3].data.cpu().numpy().tolist()
                        # ___________________________________________________       

                        print('left_top_x = ', left_top_x)                  
                        print('left_top_y = ', left_top_y)                  

                        # print(f'{left_top_x = }')
                        # print(f'{left_top_y = }')
                        # print(f'{right_down_x = }')
                        # print(f'{right_down_y = }')
                        print()

                        people_group.append([left_top_x, left_top_y, right_down_x, right_down_y])
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    else:
                        people_group.clear()
                        
                    
            # ___________________________________________________________________
            data_dict['xyxy'] = people_group
            # ___________________________________________________________________
                
            now_people = len(people_group)
            print('people_group = ', people_group)
            print('now_people = ', now_people)

            if now_people > people_max:
                people_max = now_people
            lable_count['text'] = 'Now: ' + str(now_people) + '  ' + 'Max:' + str(people_max)


            # Stream results
            # ==========================================================================================
            im0 = annotator.result()
            if view_img:
                global draw_points_list
                global draw_points_list_all
                global image_width
                global image_height

                # print(f'\n{draw_points_list_all = }') 
                # print(f'{len(draw_points_list_all) = }')

                cvimage = cv2.cvtColor(im0, cv2.COLOR_BGR2RGBA)
                pilImage = Image.fromarray(cvimage)
                
                scale_w = image_width / pilImage.width
                scale_h = image_height / pilImage.height

                cross_data = cross_detect(draw_points_list_all, people_group, scale_w, scale_h)
                area_people = cross_data[0]
                people_kepoints_group = cross_data[1]

                # print(f'\nif_view_img {area_people = }')
                # print(f'\nif_view_img {people_group = }')

                pilImage = pilImage.resize((image_width, image_height), Image.ANTIALIAS)
                dr_pilImage = ImageDraw.Draw(pilImage)
                dr_pilImage.line(draw_points_list, fill = (255, 0, 0), width=6)

                # ============================================================================
                r = 30
                if len(people_group) != 0:
                    for item in people_kepoints_group:
                        print(f'{item = }')
                        dr_pilImage.ellipse((item[0], item[1], item[0]+r, item[1]+r), fill=(255, 118, 0))

                # ============================================================================
                for i in range(len(draw_points_list_all)):
                    if area_people[i] > 0:
                        box_color = (255, 222, 0)
                    else:
                        box_color = (0, 255, 0)
                    dr_pilImage.line(draw_points_list_all[i], fill=box_color, width=6)
                # ============================================================================
                
                tkImage = ImageTk.PhotoImage(image=pilImage)
                root_window.children['!canvas'].create_image(0, 0, anchor='nw', image=tkImage)
                # root_window.children['!canvas'].

                root_window.update()
                root_window.after(1)

                # ============================================================================
                lable_text = ''
                for i in range(len(area_people)):
                    if (i+1) % 2 == 0:
                        lable_text += 'A: ' + str(i+1) + ' P: ' + str(area_people[i]) + '\n'
                    else:
                        lable_text += 'A: ' + str(i+1) + ' P: ' + str(area_people[i]) + '   '
                lable_areas['text'] = lable_text

            # ==========================================================================================


def parse_opt(new_data_dict):
    parser = argparse.ArgumentParser()

    print('parser = ', parser)
    
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')


    # _____________________________________________________________________________________________________________
    parser.add_argument('--source', type=str, default=1, help='file/dir/URL/glob, 0 for webcam')
    # parser.add_argument('--source', type=str, default='rtsp://url', help='file/dir/URL/glob, 0 for webcam')
    # _____________________________________________________________________________________________________________


    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')

    # _______________________________________________________________________________________________________________________________
    # Only detect human
    parser.add_argument('--classes', nargs='+', default=0, type=int, help='filter by class: --classes 0, or --classes 0 2 3')

    # Only detect multiple class objects
    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    # _______________________________________________________________________________________________________________________________

    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--data_dict', default=new_data_dict, help='the shared listed between the main and subprocess')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))

    print('opt = ', opt)
    return opt



# =======================================================================================================
# =======================================================================================================
def mainDetection(data_dict, a):
    opt = parse_opt(data_dict)
    print(f'\n\n{opt = }')
    run(**vars(opt))


# =======================================================================================================
def b1_move(event):
    print(f'{event.x = }')
    print(f'{event.y = }')
    global draw_points_list
    global b1_moving
    draw_points_list.append((event.x, event.y))
    if not b1_moving:
        b1_moving = True


# =======================================================================================================
def b3_press(event):
    print(f'b3_press {event.x = }')
    print(f'b3_press {event.y = }')
    global draw_points_list
    global draw_points_list_all
    draw_points_list.clear()
    draw_points_list_all.clear()



# =======================================================================================================
def b1_release(event):
    print(f'b1_release {event.x = }')
    print(f'b1_release {event.y = }')
    global b1_moving
    global draw_points_list
    global draw_points_list_all
    if b1_moving:
        draw_points_list_all.append(draw_points_list.copy())
        b1_moving = False
        draw_points_list.clear()




# =======================================================================================================
# Detection Functions
# =======================================================================================================
def get_area_max_min(area):
    x_set = []
    y_set = []
    for xy in area:
        x_set.append(xy[0])
        y_set.append(xy[1])
    return [max(x_set), min(x_set), max(y_set), min(y_set)]


# =======================================================================================================
def if_lines_cross(x11, y11, x12, y12, x21, y21, x22, y22):
    if (x11-x12) != 0:
        k1 = (y11-y12) / (x11-x12)
    else:
        k1 = 0
    if (x21-x22) != 0:
        k2 = (y21-y22) / (x21-x22)
    else:
        k2 = 0
    if k1 == k2:
        return False
    else:
        a = x12*((y11-y12)/(x11-x12)) - x22*((y21-y22)/(x21-x22)) - y12 + y22
        b = ((y11-y12)/(x11-x12)) - ((y21-y22)/(x21-x22))
        x_cross = a/b
        y_cross = ((x_cross-x12)/(x11-x12))*(y11-y12)+y12
        x1_max = max([x11, x12])
        x1_min = min([x11, x12])
        y1_max = max([y11, y12])
        y1_min = min([y11, y12])
        if (x_cross <= x1_max) and (x_cross >= x1_min) and (y_cross <= y1_max) and (y_cross >= y1_min):
            return True
        else:
            return False


# =======================================================================================================
# =======================================================================================================
def cross_detect(draw_points_list_all, people_group, scale_w, scale_h):
    people_kepoints_group = []

    # Detect key point model
    # 1: down  0: central
    # =====================
    run_model = 0
    # =====================

    for item in people_group:
        # print(f'{item = }')
        left_top_x = scale_w*item[0]
        left_top_y = scale_h*item[1]
        right_down_x = scale_w*item[2]
        right_down_y = scale_h*item[3]

        if run_model == 0:
            high_adjust = 0
            midpoint_x = int((left_top_x + right_down_x) / 2)
            midpoint_y = int((left_top_y + right_down_y) / 2 - high_adjust)
        elif run_model == 1:
            high_adjust = 50
            midpoint_x = int((left_top_x + right_down_x) / 2)
            midpoint_y = int(right_down_y - high_adjust)

        people_kepoints_group.append([midpoint_x, midpoint_y])

    area_people = []
    sum_area = len(draw_points_list_all)

    for i in range(sum_area):
        area_people.append(0)
        # print(f'{area_people = }')

    for person_coordinates in people_kepoints_group:
        # print(f'new {person_coordinates = }')

        for area_i in range(sum_area):
            unit_draw_points_list_all = draw_points_list_all[area_i]
            maxmin_xy = get_area_max_min(unit_draw_points_list_all)
            len_points_unit = len(unit_draw_points_list_all)

            # print(f'{maxmin_xy = }')

            if (person_coordinates[0] > maxmin_xy[0]) or \
                (person_coordinates[0] < maxmin_xy[1]) or \
                    (person_coordinates[1] > maxmin_xy[2]) or \
                        (person_coordinates[1] < maxmin_xy[3]):
                        # area_people[area_i] += 0
                        pass
            else:
                x21 = 0
                y21 = person_coordinates[1]
                x22 = person_coordinates[0]
                y22 = person_coordinates[1]

                area_i_people = 0

                for i in range(len_points_unit-1):
                    x11 = unit_draw_points_list_all[i][0]
                    y11 = unit_draw_points_list_all[i][1]
                    x12 = unit_draw_points_list_all[i+1][0]
                    y12 = unit_draw_points_list_all[i+1][1]

                    cross = if_lines_cross(x11, y11, x12, y12, x21, y21, x22, y22)
                    # print(f'if cross: {cross = }')

                    if cross:
                         area_i_people += 1
                    # print(f'if cross: {area_people = }')
                
                if area_i_people-1 % 2 == 0:
                    area_people[area_i] += 0
                else:
                    area_people[area_i] += 1
                
            # print(f'cross_detect: {area_people = }')
    
    return [area_people, people_kepoints_group]

# =======================================================================================================
# =======================================================================================================

root_window = tk.Tk()
root_window.title('Traffic People Detection')
# get the size of the current screen:
screenwidth = root_window.winfo_screenwidth()
screenheight = root_window.winfo_screenheight()

# Maxsize the window with the size of the screen
root_window.geometry("%dx%d" %(screenwidth, screenheight))
root_window.attributes("-topmost",True)

image_width = int(screenwidth*0.8)
image_height = int(screenheight*0.8)

# image_width = 1000
# image_height = 800

lable = Label(root_window, text='Detecting', bg='white', font=("consle", 16), width=10, height=1)
lable_count = Label(root_window, text='Runing...', bg='white', font=("consle", 16), width=25, height=1)
lable_areas = Label(root_window, text='Area People ...', bg='white', anchor='nw', font=("consle", 16), wraplength=260, justify="left", width=21, height=28)
canvas = Canvas(root_window, bg='white', width=image_width, height=image_height) 

lable.place(x=20, y=20)
lable_count.place(x=170, y=20)
lable_areas.place(x=1260, y=83)
canvas.place(x=20, y=80)

canvas.bind('<B1-Motion>', b1_move)
# canvas.bind('<ButtonPress-1>', b1_press)
canvas.bind('<ButtonPress-3>', b3_press)
canvas.bind('<ButtonRelease-1>', b1_release)

# canvas.create_line()
# ____________________________________________________________________________________________
# ____________________________________________________________________________________________
if __name__ == "__main__":

    with multiprocessing.Manager() as MG:  
        data_dict = MG.dict()
        
        p = multiprocessing.Process(target=mainDetection, args=(data_dict, 1))
        p.start()
       

        
        # while True:
        #     time.sleep(0.5)
        #     # print(f'\n ============== {data_dict = }')
        #     print('\n data_dict ============== ', data_dict)

        
        root_window.mainloop()  


