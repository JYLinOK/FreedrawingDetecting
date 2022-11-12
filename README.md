# FreedrawingDetecting
Code for paper: Multi-area Target Individual Detection with Free Drawing on Video 

## 1. Run the main.py to run the code.
## 2. Set the follow code to set the coresponding camera used to detect:

parser.add_argument('--source', type=str, default=1, help='file/dir/URL/glob, 0 for webcam')

if you have only one usb camera, set the parameter of default = 0, have two, set default = 0 / 1, and so on.

if you have web camera, set: default='rtsp://url (your web camera url)' to run your web camera.

## 3. If you only want to detect the human object, please set:

parser.add_argument('--classes', nargs='+', default=0, type=int, help='filter by class: --classes 0, or --classes 0 2 3')

if you wan to detect all classes of objects, set:

parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')








