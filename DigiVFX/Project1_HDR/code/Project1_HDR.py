import cv2
import glob
import numpy as np
import random
import matplotlib.pyplot as plt

list_img = glob.glob("../data/*.jpg")
img_num = len(list_img)
list_proccess_img = []
shutter_time = np.array([1/1250, 1/640, 1/250, 1/125, 1/64, 1/32, 1/16, 1/3])
log_shutter_time  = np.log(shutter_time).astype("float32")
for i, img_pth in enumerate(list_img):
    img = cv2.imread(img_pth)
    img = cv2.resize(img, (1024, 768))
    list_proccess_img.append({
        "image": img,
        "shutter_time": shutter_time[i],
    })
img_shape = img.shape


def shift_mtx(shift_x, shift_y):
    M = np.float32([
        [1, 0, shift_x],
        [0, 1, shift_y],
    ])
    return M

def getImageShift(src, target, shift_x, shift_y):
    h, w = target.shape[:2]
    binary_threshold = 3
    _,binary_img_src = cv2.threshold(src,np.median(src) + binary_threshold,255,cv2.THRESH_BINARY)
    binary_img_src = binary_img_src.astype(dtype=bool)
    img_compare_src = src*binary_img_src
    _,binary_img_target = cv2.threshold(target,np.median(target) + binary_threshold,255,cv2.THRESH_BINARY)
    binary_img_target = binary_img_target.astype(dtype=bool)
    img_compare_target = target*binary_img_target
    min_error = np.inf
    finetune_x, finetune_y = 0, 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            # print(i, j)
            shift_img_compare_src = cv2.warpAffine(img_compare_src, shift_mtx(shift_x + i, shift_y + j), (w, h))
            error = np.sum(np.abs(shift_img_compare_src-img_compare_target))
            if min_error > error:
                min_error = error
                
                finetune_x, finetune_y = i, j
                # print("min_error:%s in %s/%s" % (min_error, i, j))
    return shift_x + finetune_x, shift_y + finetune_y
    # return 0, 0
    


# recursive find align
def getAlignment(src, target, iter=5):
    # iter 0 means compare smallest size
    # print("iter:%s" % iter)
    if iter==0:
        shift_x, shift_y = getImageShift(src, target, 0, 0)
    else:
        down_sampling_src = cv2.pyrDown(src)
        down_sampling_target = cv2.pyrDown(target)
        # use previous shift to align
        prev_shift_x, prev_shift_y = getAlignment(down_sampling_src, down_sampling_target, iter-1)
        shift_x, shift_y = getImageShift(src, target, prev_shift_x*2, prev_shift_y*2)
        # print("result %s/%s"%(shift_x, shift_y))
    return shift_x, shift_y

    
def linear_weight(x):
    minz, maxz = 0, 255
    for i in range(len(x)):
        x[i] = maxz-x[i] if x[i]> 127.5 else x[i]
        # if x[i]>((minz+maxz)/2):
        #     x[i] = maxz-x[i]
        # else:
        #     x[i] = x[i]-minz
    return x


def sampling_z(N, list_proccess_img):
    img_num = len(list_proccess_img)
    img_shape = list_proccess_img[0]["image"].shape
    random.seed(644)
    sample_points = np.array(random.sample(range(img_shape[0]*img_shape[1]), N))
    loc_x = sample_points // img_shape[1]
    loc_y = sample_points % img_shape[1]
    sampling_z_bgr = np.array([
        [
            list_proccess_img[i]["image"][loc_x, loc_y, j] for i in range(img_num)
        ] for j in range(3)
    ])
    return sampling_z_bgr

def PaulDebevecMethod(list_proccess_img, sampling_z_bgr, N, channel_num, lambda_smooth):
    img_num = len(list_proccess_img)
    mtx_A = np.zeros((N*img_num+255, 256+N))
    mtx_B = np.zeros(N*img_num+255)
    flat_sampling_z = sampling_z_bgr[channel_num].flatten()
    weigted_zij = linear_weight(flat_sampling_z)
    mtx_A[N*img_num, 127] = 127
    mtx_A[np.arange(N*img_num), flat_sampling_z] = weigted_zij
    for i in range(img_num):
        # N nums
        mtx_A[i*N: i*N+N, 256:] = -np.identity(N) * weigted_zij[i*N: i*N+N]
        mtx_B[i*N: i*N+N] = np.log(list_proccess_img[i]["shutter_time"])
    for i in range(254):
        weight_factor = 254-i if i+1 > 127.5 else i+1
        mtx_A[N*img_num + 1 + i,i:i+3] = np.array([1, -2, 1])*lambda_smooth*weight_factor

    mtx_B[np.arange(N*img_num)] *= weigted_zij
    A_inv = np.linalg.pinv(mtx_A)
    solve_x = A_inv@mtx_B
    return solve_x[:256]

def optimize_e(sampling_z_data, G, shutter_time):
    flat_sampling_z = sampling_z_data.flatten()
    weigted_zij = linear_weight(flat_sampling_z).reshape(len(shutter_time), -1) / 128
    G_selected = G[flat_sampling_z].reshape(len(shutter_time), -1)
    numerator = 0.0
    denominator = 0.0
    for i in range(len(shutter_time)):
        numerator += G_selected[i]*weigted_zij[i]*shutter_time[i]
        denominator += weigted_zij[i]*(shutter_time[i]**2)
    Ei = numerator/(denominator+1e-7)
    return Ei.astype(np.float32)

def optimize_g(sampling_z_data, G, Ei, shutter_time):
    # Ei_shutterTime = Ei*shutter_time
    for m in range(256):
        index = np.where(sampling_z_data == m)
        len_index = len(index[0])
        sum_result = 0
        if len_index == 0:
            continue
        for idx_loc in range(len_index):
            sum_result += Ei[index[1][idx_loc]] * shutter_time[index[0][idx_loc]]
        # sum_result = np.sum(Ei_shutterTime[index]).astype(np.float32)
        # if len(index[0]) != 0 :
        
        G[m] = sum_result/len_index
    G = G/G[127]
    return G

def Robertson(sampling_z_bgr, shutter_time, num_epoch):
    g_bgr = np.array([np.arange(256)/256 for i in range(3)]).astype(np.float32)
    for idx_channel in range(3):
        sampling_z_data = sampling_z_bgr[idx_channel]
        G = g_bgr[idx_channel]
        for epoch in range(num_epoch):
            Ei = optimize_e(sampling_z_data, G, shutter_time)
            G = optimize_g(sampling_z_data, G, Ei, shutter_time)
        g_bgr[idx_channel] = G
    return np.log(g_bgr)

def smooth(data, pad_width=10):
    result = data
    pad_data = np.pad(data, pad_width, mode='edge')
    for i, loc in enumerate(range(pad_width, len(pad_data)-pad_width)):
        result[i] = np.mean(pad_data[loc-pad_width:loc+pad_width])
    return result

def get_restore_radiance(list_proccess_img, rgb_solve_g_lne):
    img_shape = list_proccess_img[0]["image"].shape
    ln_radiance_bgr = np.zeros(img_shape,dtype=np.float32)
    for c in range(img_shape[-1]):
        print(c)
        weight_sum = np.zeros((img_shape[0], img_shape[1]),dtype=np.float32)
        ln_radiance_sum = np.zeros((img_shape[0], img_shape[1]),dtype=np.float32)
        for p in range(img_num):
            img_per_channel = list_proccess_img[p]["image"][:, :, c].flatten()
            # lnE = g(Zij) - ln(shuttertime)
            ln_radiance = (rgb_solve_g_lne[c][img_per_channel] - np.log(list_proccess_img[p]["shutter_time"]).astype("float32")).reshape(img_shape[0], img_shape[1])
            # sum all weight dot radiance to get more robust result
            weight_factor = linear_weight(img_per_channel).reshape(img_shape[0], img_shape[1])
            weighted_ln_radiance = ln_radiance * weight_factor
            ln_radiance_sum += weighted_ln_radiance
            weight_sum += weight_factor
        weighted_ln_radiance = ln_radiance_sum / (weight_sum + 1e-6)
        ln_radiance_bgr[:, :, c] = weighted_ln_radiance
    return np.exp(ln_radiance_bgr)

def local_tone_mapping(bgr_radiance, delta, a):
    result = bgr_radiance
    for i in range(3):
        # delta is small value prevent singularity
        Lw_avg = np.exp(np.mean(np.log(delta + bgr_radiance[:, :, i])))
        Lm = (a / Lw_avg) * bgr_radiance[:, :, i]
        # L_white = 3
        L_white = np.max(Lm)
        Ld = (Lm * (1+(Lm/(L_white**2)))) / (1+Lm)
        result[:, :, i] = np.clip(np.array(Ld * 255), 0, 255)
    result = result.astype(np.uint8)
    # cv2.imshow("image", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return result
def global_tone_mapping(bgr_radiance, delta, a):
    # delta is small value prevent singularity
    Lw_avg = np.exp(np.mean(np.log(delta + bgr_radiance)))
    Lm = (a / Lw_avg) * bgr_radiance
    # L_white = 3
    L_white = np.max(Lm)
    Ld = (Lm * (1+(Lm/(L_white**2)))) / (1+Lm)
    result = np.clip(np.array(Ld * 255), 0, 255).astype(np.uint8)
    cv2.imshow("image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite("tonemap_photographic_global.jpg", result)
    return result

# align image 
for i in range(1, len(list_proccess_img)):
    gray_src_img = cv2.cvtColor(list_proccess_img[i-1]["image"], cv2.COLOR_BGR2GRAY)
    gray_target_img = cv2.cvtColor(list_proccess_img[i-1]["image"], cv2.COLOR_BGR2GRAY)
    shift_x, shift_y = getAlignment(gray_src_img, gray_target_img, iter=5)
    list_proccess_img[i]["image"] = cv2.warpAffine(list_proccess_img[i]["image"], shift_mtx(shift_x, shift_y), list_proccess_img[i]["image"].shape[:2][::-1])
    print("result %s/%s"%(shift_x, shift_y))

N = 1500
lambda_smooth = 20
num_epoch = 10
sampling_z_bgr = sampling_z(N, list_proccess_img)
rgb_solve_g_lne_Debevec = [PaulDebevecMethod(list_proccess_img, sampling_z_bgr, N, i, lambda_smooth) for i in range(img_shape[2])]
rgb_solve_g_lne_Robertson = Robertson(sampling_z_bgr, shutter_time, num_epoch)
smooth_log_g_bgr = rgb_solve_g_lne_Robertson
list_channel = ["blue", "green", "red"]
for i, channel_name in enumerate(list_channel):
    smooth_log_g_bgr[i] = smooth(rgb_solve_g_lne_Robertson[i], 20)
    # smooth_log_g_bgr[i] = smooth(smooth_log_g_bgr[i], 8)
    plt.plot(rgb_solve_g_lne_Debevec[i], np.arange(len(rgb_solve_g_lne_Debevec[i])))
    plt.plot(smooth_log_g_bgr[i], np.arange(len(smooth_log_g_bgr[i])))
    plt.legend(['Debevec','Robertson'])
    plt.xlabel("lnE")
    plt.ylabel("Value")
    plt.title('Compare in %s channel'%(channel_name))
    plt.show()
restore_bgr_radiance = get_restore_radiance(list_proccess_img, smooth_log_g_bgr)
cv2.imwrite('radiance_robertson.hdr', restore_bgr_radiance)
result = local_tone_mapping(restore_bgr_radiance, 1e-7, 0.25)
cv2.imwrite("Robertson_tonemapping.png", result)
