# -- coding: utf-8 --
import sys
import threading
import msvcrt
from ctypes import *
import numpy as np
import time

import cv2
#sys.path.append("C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport")
#sys.path.append("../MvImport")
from MvCameraControl_class import *
import numpy
#导入mediapipe
import mediapipe as mp



g_bExit = False
def fun_facedetect_test(image):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.1) as face_detection:

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)
        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
    return image

# 显示图像
def image_show(image):
    image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_AREA)
    cv2.imshow('test', image)
    k = cv2.waitKey(1) & 0xff

def Mono_numpy(self, data, nWidth, nHeight):
    data_ = np.frombuffer(data, count=int(nWidth * nHeight), dtype=np.uint8, offset=0)
    data_mono_arr = data_.reshape(nHeight, nWidth)
    numArray = np.zeros([nHeight, nWidth, 1], "uint8")
    numArray[:, :, 0] = data_mono_arr
    return numArray

def Color_numpy(self, data, nWidth, nHeight):
    data_ = np.frombuffer(data, count=int(nWidth * nHeight * 3), dtype=np.uint8, offset=0)
    data_r = data_[0:nWidth * nHeight * 3:3]
    data_g = data_[1:nWidth * nHeight * 3:3]
    data_b = data_[2:nWidth * nHeight * 3:3]
    data_r_arr = data_r.reshape(nHeight, nWidth)
    data_g_arr = data_g.reshape(nHeight, nWidth)
    data_b_arr = data_b.reshape(nHeight, nWidth)
    numArray = np.zeros([nHeight, nWidth, 3], "uint8")
    numArray[:, :, 0] = data_r_arr
    numArray[:, :, 1] = data_g_arr
    numArray[:, :, 2] = data_b_arr
    return numArray
# 判读图像格式是彩色还是黑白
def IsImageColor(enType):
    dates = {
        PixelType_Gvsp_RGB8_Packed: 'color',
        PixelType_Gvsp_BGR8_Packed: 'color',
        PixelType_Gvsp_YUV422_Packed: 'color',
        PixelType_Gvsp_YUV422_YUYV_Packed: 'color',
        PixelType_Gvsp_BayerGR8: 'color',
        PixelType_Gvsp_BayerRG8: 'color',
        PixelType_Gvsp_BayerGB8: 'color',
        PixelType_Gvsp_BayerBG8: 'color',
        PixelType_Gvsp_BayerGB10: 'color',
        PixelType_Gvsp_BayerGB10_Packed: 'color',
        PixelType_Gvsp_BayerBG10: 'color',
        PixelType_Gvsp_BayerBG10_Packed: 'color',
        PixelType_Gvsp_BayerRG10: 'color',
        PixelType_Gvsp_BayerRG10_Packed: 'color',
        PixelType_Gvsp_BayerGR10: 'color',
        PixelType_Gvsp_BayerGR10_Packed: 'color',
        PixelType_Gvsp_BayerGB12: 'color',
        PixelType_Gvsp_BayerGB12_Packed: 'color',
        PixelType_Gvsp_BayerBG12: 'color',
        PixelType_Gvsp_BayerBG12_Packed: 'color',
        PixelType_Gvsp_BayerRG12: 'color',
        PixelType_Gvsp_BayerRG12_Packed: 'color',
        PixelType_Gvsp_BayerGR12: 'color',
        PixelType_Gvsp_BayerGR12_Packed: 'color',
        PixelType_Gvsp_Mono8: 'mono',
        PixelType_Gvsp_Mono10: 'mono',
        PixelType_Gvsp_Mono10_Packed: 'mono',
        PixelType_Gvsp_Mono12: 'mono',
        PixelType_Gvsp_Mono12_Packed: 'mono'}
    return dates.get(enType, '未知')


# 需要显示的图像数据转换
def image_control(image,stFrameInfo):
    if stFrameInfo.enPixelType == PixelType_Gvsp_Mono8:
        image = image.reshape((stFrameInfo.nHeight,stFrameInfo.nWidth))
        image_show(image=image)#显示
    elif stFrameInfo.enPixelType == PixelType_Gvsp_BayerGB8:
        image = image.reshape((stFrameInfo.nHeight,stFrameInfo.nWidth))
        image = cv2.cvtColor(image, cv2.COLOR_BAYER_GB2RGB)
        image_show(image=image)
    elif stFrameInfo.enPixelType == PixelType_Gvsp_BayerGR8:
        image = image.reshape((stFrameInfo.nHeight,stFrameInfo.nWidth))
        image = cv2.cvtColor(image, cv2.COLOR_BAYER_GR2RGB)
        image_show(image=image)
    elif stFrameInfo.enPixelType == PixelType_Gvsp_BayerRG8:
        image = image.reshape((stFrameInfo.nHeight,stFrameInfo.nWidth))
        image = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2RGB)
        image_show(image=image)
    elif stFrameInfo.enPixelType == PixelType_Gvsp_BayerBG8:
        image = image.reshape((stFrameInfo.nHeight,stFrameInfo.nWidth))
        image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2RGB)
        image_show(image=image)
    elif stFrameInfo.enPixelType == PixelType_Gvsp_RGB8_Packed:
        image = image.reshape(stFrameInfo.nHeight,stFrameInfo.nWidth,3)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_show(image=image)
    elif stFrameInfo.enPixelType == PixelType_Gvsp_YUV422_Packed:#YUV422_8_UYVY有问题
        image = image.reshape(stFrameInfo.nHeight,stFrameInfo.nWidth,2)
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_Y422)
        image_show(image=image)
    else:
        print("Not support ImageFormat!!! \n")


#实现GetImagebuffer函数取流，HIK格式转换函数
def work_thread_1(cam=0, pData=0, nDataSize=0):
    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    print("work_thread_1!\n")
    img_buff = None
    while True:
        ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
        if ret == 0:
            print ("MV_CC_GetOneFrameTimeout: Width[%d], Height[%d], nFrameNum[%d]"  % (stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))
            time_start = time.time()
            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))
            if IsImageColor(stFrameInfo.enPixelType) == 'mono':
                print("mono!")
                stConvertParam.enDstPixelType = PixelType_Gvsp_Mono8
                nConvertSize = stFrameInfo.nWidth * stFrameInfo.nHeight
            elif IsImageColor(stFrameInfo.enPixelType) == 'color':
                print("color!")
                stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed  # opecv要用BGR，不能使用RGB
                nConvertSize = stFrameInfo.nWidth * stFrameInfo.nHeight* 3
            else:
                print("not support!!!")
            if img_buff is None:
                img_buff = (c_ubyte * stFrameInfo.nFrameLen)()
            # ---
            stConvertParam.nWidth = stFrameInfo.nWidth
            stConvertParam.nHeight = stFrameInfo.nHeight
            stConvertParam.pSrcData = cast(pData, POINTER(c_ubyte))
            stConvertParam.nSrcDataLen = stFrameInfo.nFrameLen
            stConvertParam.enSrcPixelType = stFrameInfo.enPixelType
            stConvertParam.pDstBuffer = (c_ubyte * nConvertSize)()
            stConvertParam.nDstBufferSize = nConvertSize
            ret = cam.MV_CC_ConvertPixelType(stConvertParam)
            if ret != 0:
                print("convert pixel fail! ret[0x%x]" % ret)
                del stConvertParam.pSrcData
                sys.exit()
            else:
                print("convert ok!!")
                # 转OpenCV
                # 黑白处理
                if IsImageColor(stFrameInfo.enPixelType) == 'mono':
                    img_buff = (c_ubyte * stConvertParam.nDstLen)()
                    cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pDstBuffer, stConvertParam.nDstLen)
                    img_buff = np.frombuffer(img_buff,count=int(stConvertParam.nDstLen), dtype=np.uint8)
                    img_buff = img_buff.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
                    print("mono ok!!")
                    image_show(image=img_buff)  # 显示图像函数
                # 彩色处理
                if IsImageColor(stFrameInfo.enPixelType) == 'color':
                    img_buff = (c_ubyte * stConvertParam.nDstLen)()
                    cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pDstBuffer, stConvertParam.nDstLen)
                    img_buff = np.frombuffer(img_buff, count=int(stConvertParam.nDstBufferSize), dtype=np.uint8)
                    img_buff = img_buff.reshape(stFrameInfo.nHeight,stFrameInfo.nWidth,3)
                    print("color ok!!")
                    image = img_buff
                    image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_AREA)
                    image = fun_facedetect_test(image)
                    #image_show(image=img_buff)  # 显示图像函数
                    #image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_AREA)
                    cv2.imshow('test22', image)
                    if cv2.waitKey(5) & 0xFF == 27:
                        break
                time_end = time.time()
                print('time cos:', time_end - time_start, 's')
        else:
            print ("no data[0x%x]" % ret)
        if g_bExit == True:
                break

#实现MV_CC_GetImageBuffer函数取流，HIK格式转换函数
def work_thread_2(cam=0, pData=0, nDataSize=0):
    #global img_buff
    img_buff = None
    stOutFrame = MV_FRAME_OUT()
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))
    while True:
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        if None != stOutFrame.pBufAddr and 0 == ret:
            print ("MV_CC_GetImageBuffer: Width[%d], Height[%d], nFrameNum[%d]"  % (stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nFrameNum))
            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))
            if IsImageColor(stOutFrame.stFrameInfo.enPixelType) == 'mono':
                print("mono!")
                stConvertParam.enDstPixelType = PixelType_Gvsp_Mono8
                nConvertSize = stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight
            elif IsImageColor(stOutFrame.stFrameInfo.enPixelType) == 'color':
                print("color!")
                stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed  # opecv要用BGR，不能使用RGB
                nConvertSize = stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight * 3
            else:
                print("not support!!!")
            if img_buff is None:
                img_buff = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
            stConvertParam.nWidth = stOutFrame.stFrameInfo.nWidth
            stConvertParam.nHeight = stOutFrame.stFrameInfo.nHeight
            stConvertParam.pSrcData = cast(stOutFrame.pBufAddr, POINTER(c_ubyte))
            stConvertParam.nSrcDataLen = stOutFrame.stFrameInfo.nFrameLen
            stConvertParam.enSrcPixelType = stOutFrame.stFrameInfo.enPixelType
            stConvertParam.pDstBuffer = (c_ubyte * nConvertSize)()
            stConvertParam.nDstBufferSize = nConvertSize
            ret = cam.MV_CC_ConvertPixelType(stConvertParam)
            if ret != 0:
                print("convert pixel fail! ret[0x%x]" % ret)
                del stConvertParam.pSrcData
                sys.exit()
            else:
                print("convert ok!!")
                # # 存raw图看看转化成功没有
                # file_path = "AfterConvert_RGB.raw"
                # file_open = open(file_path.encode('ascii'), 'wb+')
                # try:
                #     image_save= (c_ubyte * stConvertParam.nDstBufferSize)()
                #     cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pDstBuffer, stConvertParam.nDstLen)
                #     file_open.write(img_buff)
                #     print("raw ok!!")
                # except:
                #     raise Exception("save file executed failed:%s" % e.message)
                # finally:
                #     file_open.close()
            # 黑白处理
            if IsImageColor(stOutFrame.stFrameInfo.enPixelType) == 'mono':
                img_buff = (c_ubyte * stConvertParam.nDstLen)()
                cdll.msvcrt.memcpy(byref(img_buff),stConvertParam.pDstBuffer,stConvertParam.nDstLen)
                img_buff = np.frombuffer(img_buff, count=int(stConvertParam.nDstBufferSize), dtype=np.uint8)
                img_buff = img_buff.reshape((stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth))
                print("mono ok!!")
                image_show(image=img_buff)  # 显示图像函数
            # 彩色处理
            if IsImageColor(stOutFrame.stFrameInfo.enPixelType) == 'color':
                img_buff = (c_ubyte * stConvertParam.nDstLen)()
                cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pDstBuffer, stConvertParam.nDstLen)
                img_buff = np.frombuffer(img_buff, count=int(stConvertParam.nDstBufferSize), dtype=np.uint8)#data以流的形式读入转化成ndarray对象
                img_buff = img_buff.reshape(stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth,3)
                print("color ok!!")
                image_show(image=img_buff)  # 显示图像函数
            else:
                print("no data[0x%x]" % ret)
        nRet = cam.MV_CC_FreeImageBuffer(stOutFrame)
        if g_bExit == True:
            break

#实现getoneframe函数取流，OpenCV自带格式转换函数
def work_thread_3(cam=0, pData=0, nDataSize=0):
    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    data_buf = None
    print("work_thread_3!\n")
    #image = numpy.zeros((640, 480), dtype=numpy.uint8)
    while True:
        ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
        if ret == 0:
            print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
            stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))
            img_buff = (c_ubyte * stFrameInfo.nFrameLen)()
            cdll.msvcrt.memcpy(byref(img_buff),pData, stFrameInfo.nFrameLen)
            img_buff = np.frombuffer(img_buff, count=int(stFrameInfo.nFrameLen), dtype=np.uint8) #data以流的形式读入转化成ndarray对象
            image_control(img_buff, stFrameInfo)
        if g_bExit == True:
            break

#回调函数
#实现MV_CC_GetImageBuffer函数取流，HIK格式转换函数
winfun_ctype = WINFUNCTYPE
stFrameInfo = POINTER(MV_FRAME_OUT_INFO_EX)
pData = POINTER(c_ubyte)
FrameInfoCallBack = winfun_ctype(None, pData, stFrameInfo, c_void_p)
def image_callback(pData, pFrameInfo, pUser):
        #global img_buff
        img_buff = None
        stFrameInfo = cast(pFrameInfo, POINTER(MV_FRAME_OUT_INFO_EX)).contents
        if stFrameInfo:
            print ("callback:get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))
            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))
            if IsImageColor(stFrameInfo.enPixelType) == 'mono':
                print("mono!")
                stConvertParam.enDstPixelType = PixelType_Gvsp_Mono8
                nConvertSize = stFrameInfo.nWidth * stFrameInfo.nHeight
            elif IsImageColor(stFrameInfo.enPixelType) == 'color':
                print("color!")
                stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed  # opecv要用BGR，不能使用RGB
                nConvertSize = stFrameInfo.nWidth * stFrameInfo.nHeight * 3
            else:
                print("not support!!!")
            if img_buff is None:
                img_buff = (c_ubyte * stFrameInfo.nFrameLen)()
            # ---
            stConvertParam.nWidth = stFrameInfo.nWidth
            stConvertParam.nHeight = stFrameInfo.nHeight
            stConvertParam.pSrcData = cast(pData, POINTER(c_ubyte))
            stConvertParam.nSrcDataLen = stFrameInfo.nFrameLen
            stConvertParam.enSrcPixelType = stFrameInfo.enPixelType
            stConvertParam.pDstBuffer = (c_ubyte * nConvertSize)()
            stConvertParam.nDstBufferSize = nConvertSize
            ret = cam.MV_CC_ConvertPixelType(stConvertParam)
            if ret != 0:
                print("convert pixel fail! ret[0x%x]" % ret)
                del stConvertParam.pSrcData
                sys.exit()
            else:
                print("convert ok!!")
                # 转OpenCV
                # 黑白处理
                if IsImageColor(stFrameInfo.enPixelType) == 'mono':
                    img_buff = (c_ubyte * stConvertParam.nDstLen)()
                    cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pDstBuffer, stConvertParam.nDstLen)
                    img_buff = np.frombuffer(img_buff, count=int(stConvertParam.nDstBufferSize), dtype=np.uint8)
                    img_buff = img_buff.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
                    print("mono ok!!")
                    image_show(image=img_buff)  # 显示图像函数
                # 彩色处理
                if IsImageColor(stFrameInfo.enPixelType) == 'color':
                    img_buff = (c_ubyte * stConvertParam.nDstLen)()
                    cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pDstBuffer, stConvertParam.nDstLen)
                    img_buff = np.frombuffer(img_buff, count=int(stConvertParam.nDstBufferSize), dtype=np.uint8)
                    img_buff = img_buff.reshape(stFrameInfo.nHeight,stFrameInfo.nWidth,3)
                    print("color ok!!")
                    image_show(image=img_buff)  # 显示图像函数
CALL_BACK_FUN = FrameInfoCallBack(image_callback)
if __name__ == "__main__":

    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

    # ch:枚举设备 | en:Enum device
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        print ("enum devices fail! ret[0x%x]" % ret)
        sys.exit()

    if deviceList.nDeviceNum == 0:
        print ("find no device!")
        sys.exit()

    print ("Find %d devices!" % deviceList.nDeviceNum)

    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            print("gige device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)
            nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print("u3v device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)

            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print ("user serial number: %s" % strSerialNumber)

    nConnectionNum = input("please input the number of the device to connect:")

    if int(nConnectionNum) >= deviceList.nDeviceNum:
        print ("intput error!")
        sys.exit()

    # ch:创建相机实例 | en:Creat Camera Object
    cam = MvCamera()

    # ch:选择设备并创建句柄 | en:Select device and create handle
    stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print ("create handle fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:打开设备 | en:Open device
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print ("open device fail! ret[0x%x]" % ret)
        sys.exit()
    ret = cam.MV_CC_SetIntValue("GevHeartbeatTimeout", 5000)
    if ret != 0:
        print("Warning: Set GevHeartbeatTimeout  fail! ret[0x%x]" % ret)

     # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
    if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
        nPacketSize = cam.MV_CC_GetOptimalPacketSize()
        if int(nPacketSize) > 0:
            ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize",nPacketSize)
            if ret != 0:
                print ("Warning: Set Packet Size fail! ret[0x%x]" % ret)
        else:
            print ("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

    stBool = c_bool(False)
    ret =cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)
    if ret != 0:
        print ("get AcquisitionFrameRateEnable fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:设置触发模式为off | en:Set trigger mode as off
    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    if ret != 0:
        print ("set trigger mode fail! ret[0x%x]" % ret)
        sys.exit()
    stParam = MVCC_INTVALUE()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
    ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
    if ret != 0:
        print("get payload size fail! ret[0x%x]" % ret)
        sys.exit()
    nPayloadSize = stParam.nCurValue

    # # ch:注册抓图回调 | en:Register image callback
    # ret = cam.MV_CC_RegisterImageCallBackEx(CALL_BACK_FUN, None)
    # if ret != 0:
    #     print("register image callback fail! ret[0x%x]" % ret)
    #     sys.exit()
    # else:
    #     print("Start callback grab!!! ")

    # ch:开始取流 | en:Start grab image
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print ("start grabbing fail! ret[0x%x]" % ret)
        sys.exit()
    # 设置 bufffer 大小
    data_buffer = (c_ubyte * nPayloadSize)()
    #单帧取流函数，主动取流方法实现
    try:
        #work_thread_1 #实现GetImagebuffer函数取流，HIK格式转换函数
        #work_thread_2 #实现MV_CC_GetImageBuffer函数取流，HIK格式转换函数
        #work_thread_3 #实现getoneframe函数取流，OpenCV自带格式转换函数
        hThreadHandle = threading.Thread(target=work_thread_1,args=(cam, byref(data_buffer), nPayloadSize))
        hThreadHandle.start()
    except:
        print ("error: unable to start thread")

    print ("press a key to stop grabbing.")
    msvcrt.getch()
    g_bExit = True

    hThreadHandle.join()

    # ch:停止取流 | en:Stop grab image
    ret = cam.MV_CC_StopGrabbing()
    if ret != 0:
        print ("stop grabbing fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:关闭设备 | Close device
    ret = cam.MV_CC_CloseDevice()
    if ret != 0:
        print ("close deivce fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:销毁句柄 | Destroy handle
    ret = cam.MV_CC_DestroyHandle()
    if ret != 0:
        print ("destroy handle fail! ret[0x%x]" % ret)
        sys.exit()
