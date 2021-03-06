#test_FaceDict.py --gpu_ids -1
import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_crop
from util import html
import numpy as np
import math
from PIL import Image
import torchvision.transforms as transforms
import torch
import random
import cv2
import dlib
from skimage import transform as trans
from skimage import io
from data.image_folder import make_dataset
import sys
import youtube_dl
sys.path.append('FaceLandmarkDetection')
import face_alignment
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import time


################# functions of crop and align face images #################

def get_5_points(img):
    dets = detector(img, 1)
    if len(dets) == 0:
        return None
    areas = []
    if len(dets) > 1:
        print('\t###### Warning: more than one face is detected. In this version, we only handle the largest one.')
    for i in range(len(dets)):
        area = (dets[i].rect.right()-dets[i].rect.left())*(dets[i].rect.bottom()-dets[i].rect.top())
        areas.append(area)
    ins = areas.index(max(areas))
    shape = sp(img, dets[ins].rect)
    single_points = []
    for i in range(5):
        single_points.append([shape.part(i).x, shape.part(i).y])
    return np.array(single_points)

def align_and_save(img_path, save_path, save_input_path, save_param_path, upsample_scale=2):
    out_size = (512, 512)
    img = dlib.load_rgb_image(img_path)
    h,w,_ = img.shape
    source = get_5_points(img)
    if source is None: #
        print('\t################ No face is detected')
        return
    tform = trans.SimilarityTransform()
    tform.estimate(source, reference)
    M = tform.params[0:2,:]
    crop_img = cv2.warpAffine(img, M, out_size)
    io.imsave(save_path, crop_img) #save the crop and align face
    io.imsave(save_input_path, img) #save the whole input image
    tform2 = trans.SimilarityTransform()
    tform2.estimate(reference, source*upsample_scale)
    # inv_M = cv2.invertAffineTransform(M)
    np.savetxt(save_param_path, tform2.params[0:2,:],fmt='%.3f') #save the inverse affine parameters

def reverse_align(input_path, face_path, param_path, save_path, upsample_scale=2):
    out_size = (512, 512)
    input_img = dlib.load_rgb_image(input_path)
    h,w,_ = input_img.shape
    face512 = dlib.load_rgb_image(face_path)
    inv_M = np.loadtxt(param_path)
    inv_crop_img = cv2.warpAffine(face512, inv_M, (w*upsample_scale,h*upsample_scale))
    mask = np.ones((512, 512, 3), dtype=np.float32) #* 255
    inv_mask = cv2.warpAffine(mask, inv_M, (w*upsample_scale,h*upsample_scale))
    upsample_img = cv2.resize(input_img, (w*upsample_scale, h*upsample_scale))
    inv_mask_erosion_removeborder = cv2.erode(inv_mask, np.ones((2 * upsample_scale, 2 * upsample_scale), np.uint8))# to remove the black border
    inv_crop_img_removeborder = inv_mask_erosion_removeborder * inv_crop_img
    total_face_area = np.sum(inv_mask_erosion_removeborder)//3
    w_edge = int(total_face_area ** 0.5) // 20 #compute the fusion edge based on the area of face
    erosion_radius = w_edge * 2
    inv_mask_center = cv2.erode(inv_mask_erosion_removeborder, np.ones((erosion_radius, erosion_radius), np.uint8))
    blur_size = w_edge * 2
    inv_soft_mask = cv2.GaussianBlur(inv_mask_center,(blur_size + 1, blur_size + 1),0)
    merge_img = inv_soft_mask * inv_crop_img_removeborder + (1 - inv_soft_mask) * upsample_img
    io.imsave(save_path, merge_img.astype(np.uint8))

################ functions of preparing the test images ###################

def AddUpSample(img):
    return img.resize((512, 512), Image.BICUBIC)

def get_part_location(partpath, imgname):
    Landmarks = []
    if not os.path.exists(os.path.join(partpath,imgname+'.txt')):
        print(os.path.join(partpath,imgname+'.txt'))
        print('\t################ No landmark file')
        return 0
    with open(os.path.join(partpath,imgname+'.txt'),'r') as f:
        for line in f:
            tmp = [np.float(i) for i in line.split(' ') if i != '\n']
            Landmarks.append(tmp)
    Landmarks = np.array(Landmarks)
    Map_LE = list(np.hstack((range(17,22), range(36,42))))
    Map_RE = list(np.hstack((range(22,27), range(42,48))))
    Map_NO = list(range(29,36))
    Map_MO = list(range(48,68))
    try:
        #left eye
        Mean_LE = np.mean(Landmarks[Map_LE],0)
        L_LE = np.max((np.max(np.max(Landmarks[Map_LE],0) - np.min(Landmarks[Map_LE],0))/2,16))
        Location_LE = np.hstack((Mean_LE - L_LE + 1, Mean_LE + L_LE)).astype(int)
        #right eye
        Mean_RE = np.mean(Landmarks[Map_RE],0)
        L_RE = np.max((np.max(np.max(Landmarks[Map_RE],0) - np.min(Landmarks[Map_RE],0))/2,16))
        Location_RE = np.hstack((Mean_RE - L_RE + 1, Mean_RE + L_RE)).astype(int)
        #nose
        Mean_NO = np.mean(Landmarks[Map_NO],0)
        L_NO = np.max((np.max(np.max(Landmarks[Map_NO],0) - np.min(Landmarks[Map_NO],0))/2,16))
        Location_NO = np.hstack((Mean_NO - L_NO + 1, Mean_NO + L_NO)).astype(int)
        #mouth
        Mean_MO = np.mean(Landmarks[Map_MO],0)
        L_MO = np.max((np.max(np.max(Landmarks[Map_MO],0) - np.min(Landmarks[Map_MO],0))/2,16))
        Location_MO = np.hstack((Mean_MO - L_MO + 1, Mean_MO + L_MO)).astype(int)
    except:
        return 0
    return torch.from_numpy(Location_LE).unsqueeze(0), torch.from_numpy(Location_RE).unsqueeze(0), torch.from_numpy(Location_NO).unsqueeze(0), torch.from_numpy(Location_MO).unsqueeze(0)

def obtain_inputs(img_path, Landmark_path, img_name):
    A_paths = os.path.join(img_path,img_name)
    A = Image.open(A_paths).convert('RGB')
    Part_locations = get_part_location(Landmark_path, img_name)
    if Part_locations == 0:
        return 0
    C = A
    A = AddUpSample(A)
    A = transforms.ToTensor()(A)
    C = transforms.ToTensor()(C)
    A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A) #
    C = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(C) #
    return {'A':A.unsqueeze(0), 'C':C.unsqueeze(0), 'A_paths': A_paths,'Part_locations': Part_locations}

#def process_video(source_url, target_start, target_end):
if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    opt.which_epoch = 'latest'


        #os.system('cls||clear')
        ########################### Test Param ################################
    gpu_ids = [] # gpu id. if use cpu, set gpu_ids = []
    UpScaleWhole = 4  # the upsamle scale. It should be noted that our face results are fixed to 512.
        #TestImgPath = opt.test_path
        #ResultsDir = opt.results_dir
        #UpScaleWhole = opt.upscale_factor

    '''
    source_url=sys.argv[1]
    target_start = sys.argv[2]
    target_end = sys.argv[3]
    print(source_url, target_start, target_end)
    '''

    if True:
        TestImgPath = './TestData/TestVideo' # test video path
        ResultsDir = './Results/TestVideoResults' #save path
        print('\n###################### Now Running the {} task ##############################'.format(6))

        print('\n####################### Step 1: Download and crop video. Splitting into frames ###########################\n')

        #You can paste the YouTube link here
        print('Enter YouTube link for video: ')
        #source_url = input()
        source_url = opt.url#'https://www.youtube.com/watch?v=si-thUvEvls'
        target_start = opt.time1#'00:00:34'
        target_end = opt.time2#'00:00:36'
        '''
        if source_url == '':
            uploaded = files.upload()
            for fn in uploaded.keys():
                print('User uploaded file "{name}" with length {length} bytes'.format(name=fn, length=len(uploaded[fn])))
            file_name = "TestData/TestVideo/downloaded_video." + fn.split(".")[-1]
        else:
            try:
                ydl_opts = {
                    #'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
                    'format': 'bestvideo+audio/mp4',
                    'outtmpl': 'TestData/TestVideo/downloaded_video.mp4',
                    }
                with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([source_url])
                file_name = 'TestData/TestVideo/downloaded_video.mp4'
            except BaseException:
                fn = source_url.split('/')[-1]
                file_name = "TestData/TestVideo/downloaded_video." + fn.split(".")[-1]

        #Crop video (h:m:s)
        print('\nEnter time for cropping video: start (h:m:s), end (h:m:s)\nP.s. start = 00:00:00 and end = 00:00:00 its full video\n')
        #target_start = input()
        #target_end = input()

        if target_end != '00:00:00':
            dates1 = target_start.split(':')
            dates2 = target_end.split(':')
            target_start = int(dates1[0])*60*60 + int(dates1[1])*60 + int(dates1[2])
            target_end = int(dates2[0])*60*60 + int(dates2[1])*60 + int(dates2[2])
            SaveInputPath = os.path.join(ResultsDir,'Step1_Cropping')
            if not os.path.exists(SaveInputPath):
                os.makedirs(SaveInputPath)
            new_file_name = SaveInputPath+'/crop_downloaded_video.mp4'
            ffmpeg_extract_subclip(file_name, target_start, target_end, targetname=new_file_name)
            os.remove(file_name)
            file_name = new_file_name


        SaveFramesPath = os.path.join(ResultsDir,'Step1_Frames')
        if not os.path.exists(SaveFramesPath):
            os.makedirs(SaveFramesPath)


        vidcap = cv2.VideoCapture(file_name)
        success, image = vidcap.read()
        count = 0
        success = True
        while success:
            cv2.imwrite(SaveFramesPath+"/frame%09d.jpg" % count, image)
            success,image = vidcap.read()
            count += 1
        try:
            fps_of_video = int(cv2.VideoCapture(file_name).get(cv2.CAP_PROP_FPS))
            frames_of_video = int(cv2.VideoCapture(file_name).get(cv2.CAP_PROP_FRAME_COUNT))
            print("Video uploaded. Number of frames: {}.".format(str(count)))
        except:
            print("Video uploaded!\n")


        ###########Step 2: Crop and Align Face from the whole Image ###########
        print('\n####################### Step 2: Crop and Align Face ###########################\n')
        detector = dlib.cnn_face_detection_model_v1('./packages/mmod_human_face_detector.dat')
        sp = dlib.shape_predictor('./packages/shape_predictor_5_face_landmarks.dat')
        reference = np.load('./packages/FFHQ_template.npy') / 2

        SaveCropPath = os.path.join(ResultsDir,'Step2_CropImg')
        if not os.path.exists(SaveCropPath):
            os.makedirs(SaveCropPath)
        SaveParamPath = os.path.join(ResultsDir,'Step2_AffineParam') #save the inverse affine parameters
        if not os.path.exists(SaveParamPath):
            os.makedirs(SaveParamPath)


        ImgPaths = make_dataset(SaveFramesPath)


        for i, ImgPath in enumerate(ImgPaths):
            ImgName = os.path.split(ImgPath)[-1]
            #print('Crop and Align {} image'.format(ImgName))

            SavePath = os.path.join(SaveCropPath,ImgName)
            SaveInput = os.path.join(SaveFramesPath,ImgName)
            SaveParam = os.path.join(SaveParamPath, ImgName+'.npy')
            align_and_save(ImgPath, SavePath, SaveInput, SaveParam, UpScaleWhole)


        ####### Step 3: Face Landmark Detection from the Cropped Image ########
        print('\n####################### Step 3: Face Landmark Detection #######################\n')

        SaveLandmarkPath = os.path.join(ResultsDir,'Step3_Landmarks')

        if len(gpu_ids) > 0:
            dev = 'cuda:{}'.format(gpu_ids[0])
        else:
            dev = 'cpu'
        FD = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,device=dev, flip_input=False)
        if not os.path.exists(SaveLandmarkPath):
            os.makedirs(SaveLandmarkPath)

        ImgPaths = make_dataset(SaveCropPath)



        for i,ImgPath in enumerate(ImgPaths):
            ImgName = os.path.split(ImgPath)[-1]
            #print('Detecting {}'.format(ImgName))

            Img = io.imread(ImgPath)
            try:
                PredsAll = FD.get_landmarks(Img)
            except:
                print('\t################ Error in face detection, continue...')
                continue
            if PredsAll is None:
                print('\t################ No face, continue...')
                continue
            ins = 0
            if len(PredsAll)!=1:
                hights = []
                for l in PredsAll:
                    hights.append(l[8,1] - l[19,1])
                ins = hights.index(max(hights))
                # print('\t################ Warning: Detected too many face, only handle the largest one...')
                # continue
            preds = PredsAll[ins]
            AddLength = np.sqrt(np.sum(np.power(preds[27][0:2]-preds[33][0:2],2)))
            SaveName = ImgName+'.txt'
            np.savetxt(os.path.join(SaveLandmarkPath,SaveName),preds[:,0:2],fmt='%.3f')
            '''

        SaveLandmarkPath = os.path.join(ResultsDir,'Step3_Landmarks')
        SaveCropPath = os.path.join(ResultsDir,'Step2_CropImg')
        SaveParamPath = os.path.join(ResultsDir,'Step2_AffineParam')
        SaveFramesPath = os.path.join(ResultsDir,'Step1_Frames')

        ####################### Step 4: Face Restoration ######################
        print('\n####################### Step 4: Face Restoration ##############################\n')
        SaveRestorePath = os.path.join(ResultsDir,'Step4_RestoreCropFace')# Only Face Results
        if not os.path.exists(SaveRestorePath):
            os.makedirs(SaveRestorePath)
        model = create_model(opt)
        model.setup(opt)
        ImgPaths = make_dataset(SaveCropPath)
        total = 0



        for i, ImgPath in enumerate(ImgPaths):
            ImgName = os.path.split(ImgPath)[-1]
            #print('Restoring {}'.format(ImgName))

            data = obtain_inputs(SaveCropPath, SaveLandmarkPath, ImgName)
            if data == 0:
                print('\t################ Error in landmark file, continue...')
                continue
            total = total + 1
            model.set_input(data)
            try:
                model.test()
                visuals = model.get_current_visuals()
                save_crop(visuals,os.path.join(SaveRestorePath,ImgName))
            except Exception as e:
                print('\t################ Error in enhancing this image: {}'.format(str(e)))
                print('\t################ continue...')
                continue


        print('\n############### Step 5: Paste the Restored Face to the Input Image ############\n')
        SaveRestorePath = os.path.join(ResultsDir,'Step4_RestoreCropFace')
        SaveFinalPath = os.path.join(ResultsDir,'Step5_FinalFrames')
        SaveParamPath = os.path.join(ResultsDir,'Step2_AffineParam') #save the inverse affine parameters
        if not os.path.exists(SaveFinalPath):
            os.makedirs(SaveFinalPath)
        ImgPaths = make_dataset(SaveRestorePath)



        for i,ImgPath in enumerate(ImgPaths):
            ImgName = os.path.split(ImgPath)[-1]
            #print('Final Restoring {}'.format(ImgName))

            WholeInputPath = os.path.join('.\Results\TestVideoResults\Step1_Frames', ImgName)
            FaceResultPath = os.path.join(SaveRestorePath, ImgName)
            ParamPath = os.path.join(SaveParamPath, ImgName+'.npy')
            SaveWholePath = os.path.join(SaveFinalPath, ImgName)
            reverse_align(WholeInputPath, FaceResultPath, ParamPath, SaveWholePath, UpScaleWhole)

        print('Done!')

        print('\n####################### Step 6: Merging frames into a new video ###########################\n')
        path_orig_frame = './Results/TestVideoResults/Step1_Frames/'
        path_to_img = './Results/TestVideoResults/Step5_FinalFrames/'
        if len(os.listdir(path_to_img)) == 0:
            path_to_img = './Results/TestVideoResults/Step4_RestoreCropFace/'
        final_video = './Results/TestVideoResults/Step6_FinalVideo/'
        if not os.path.exists(final_video):
            os.makedirs(final_video)
        img = os.listdir(path_to_img)
        orig_img = os.listdir(path_orig_frame)
        orig_img.sort()
        staffs = []
        fps_of_video = int(cv2.VideoCapture('./Results/TestVideoResults/Step1_Cropping/crop_downloaded_video.mp4').get(cv2.CAP_PROP_FPS))



        for i in img:
            if os.path.isfile(path_to_img + i):
                staffs.append(path_to_img + i)
            else:
                staffs.append(path_orig_frame + i)
            staff = cv2.imread(staffs[0])  # get size from the 1st frame
            name_video = final_video+'result.mp4'

            writer = cv2.VideoWriter(
                name_video,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps_of_video,
                (staff.shape[1], staff.shape[0]),  # width, height
                isColor=len(staff.shape) > 2)
            for staff in map(cv2.imread, staffs):
                writer.write(staff)
            writer.release()

        print('Done!')
        print('\nRemastering video did success, video name: result.mp4')

    '''
    if answer=='2':
        TestImgPath = './TestData/TestWhole' # test image path
        ResultsDir = './Results/TestWholeResults' #save path
        print('\n###################### Now Running the {} task ##############################'.format(UpScaleWhole))

        ###########Step 1: Crop and Align Face from the whole Image ###########
        print('\n####################### Step 1: Crop and Align Face ###########################\n')

        detector = dlib.cnn_face_detection_model_v1('./packages/mmod_human_face_detector.dat')
        sp = dlib.shape_predictor('./packages/shape_predictor_5_face_landmarks.dat')
        reference = np.load('./packages/FFHQ_template.npy') / 2
        SaveInputPath = os.path.join(ResultsDir,'Step0_Input')
        if not os.path.exists(SaveInputPath):
            os.makedirs(SaveInputPath)
        SaveCropPath = os.path.join(ResultsDir,'Step1_CropImg')
        if not os.path.exists(SaveCropPath):
            os.makedirs(SaveCropPath)

        SaveParamPath = os.path.join(ResultsDir,'Step1_AffineParam') #save the inverse affine parameters
        if not os.path.exists(SaveParamPath):
            os.makedirs(SaveParamPath)

        ImgPaths = make_dataset(TestImgPath)
        for i, ImgPath in enumerate(ImgPaths):
            ImgName = os.path.split(ImgPath)[-1]
            print('Crop and Align {} image'.format(ImgName))
            SavePath = os.path.join(SaveCropPath,ImgName)
            SaveInput = os.path.join(SaveInputPath,ImgName)
            SaveParam = os.path.join(SaveParamPath, ImgName+'.npy')
            align_and_save(ImgPath, SavePath, SaveInput, SaveParam, UpScaleWhole)

        ####### Step 2: Face Landmark Detection from the Cropped Image ########
        print('\n####################### Step 2: Face Landmark Detection #######################\n')

        SaveLandmarkPath = os.path.join(ResultsDir,'Step2_Landmarks')
        if len(gpu_ids) > 0:
            dev = 'cuda:{}'.format(gpu_ids[0])
        else:
            dev = 'cpu'
        FD = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,device=dev, flip_input=False)
        if not os.path.exists(SaveLandmarkPath):
            os.makedirs(SaveLandmarkPath)
        ImgPaths = make_dataset(SaveCropPath)
        for i,ImgPath in enumerate(ImgPaths):
            ImgName = os.path.split(ImgPath)[-1]
            print('Detecting {}'.format(ImgName))
            Img = io.imread(ImgPath)
            try:
                PredsAll = FD.get_landmarks(Img)
            except:
                print('\t################ Error in face detection, continue...')
                continue
            if PredsAll is None:
                print('\t################ No face, continue...')
                continue
            ins = 0
            if len(PredsAll)!=1:
                hights = []
                for l in PredsAll:
                    hights.append(l[8,1] - l[19,1])
                ins = hights.index(max(hights))
                # print('\t################ Warning: Detected too many face, only handle the largest one...')
                # continue
            preds = PredsAll[ins]
            AddLength = np.sqrt(np.sum(np.power(preds[27][0:2]-preds[33][0:2],2)))
            SaveName = ImgName+'.txt'
            np.savetxt(os.path.join(SaveLandmarkPath,SaveName),preds[:,0:2],fmt='%.3f')

        ####################### Step 3: Face Restoration ######################
        print('\n####################### Step 3: Face Restoration ##############################\n')

        SaveRestorePath = os.path.join(ResultsDir,'Step3_RestoreCropFace')# Only Face Results
        if not os.path.exists(SaveRestorePath):
            os.makedirs(SaveRestorePath)
        model = create_model(opt)
        model.setup(opt)
        ImgPaths = make_dataset(SaveCropPath)
        total = 0
        for i, ImgPath in enumerate(ImgPaths):
            ImgName = os.path.split(ImgPath)[-1]
            print('Restoring {}'.format(ImgName))
            data = obtain_inputs(SaveCropPath, SaveLandmarkPath, ImgName)
            if data == 0:
                print('\t################ Error in landmark file, continue...')
                continue #
            total = total + 1
            model.set_input(data)
            try:
                model.test()
                visuals = model.get_current_visuals()
                save_crop(visuals,os.path.join(SaveRestorePath,ImgName))
            except Exception as e:
                print('\t################ Error in enhancing this image: {}'.format(str(e)))
                print('\t################ continue...')
                continue

        print('\n############### Step 4: Paste the Restored Face to the Input Image ############\n')
        SaveRestorePath = os.path.join(ResultsDir,'Step3_RestoreCropFace')# Only Face Results
        SaveFinalPath = os.path.join(ResultsDir,'Step4_FinalResults')
        SaveParamPath = os.path.join(ResultsDir,'Step1_AffineParam') #save the inverse affine parameters
        if not os.path.exists(SaveFinalPath):
            os.makedirs(SaveFinalPath)
        ImgPaths = make_dataset(SaveRestorePath)
        for i,ImgPath in enumerate(ImgPaths):
            ImgName = os.path.split(ImgPath)[-1]
            print('Final Restoring {}'.format(ImgName))
            WholeInputPath = os.path.join(TestImgPath,ImgName)
            FaceResultPath = os.path.join(SaveRestorePath, ImgName)
            ParamPath = os.path.join(SaveParamPath, ImgName+'.npy')
            SaveWholePath = os.path.join(SaveFinalPath, ImgName)
            reverse_align(WholeInputPath, FaceResultPath, ParamPath, SaveWholePath, UpScaleWhole)
        print('Done!')
        '''
