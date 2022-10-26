import functools
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2 import INTER_NEAREST, DescriptorMatcher, KeyPoint


class SIFT(object):
    def __init__(
            self, raw_image, sigma=1.6, assume_blur=0.5, num_intervals=2,
            image_border_width=5, contrast_threshold=0.04):
        self.raw_image = raw_image
        self.sigma = sigma
        self.assume_blur = assume_blur
        self.num_intervals = num_intervals
        self.image_border_width = image_border_width
        self.contrast_threshold = contrast_threshold

    def GetBaseImg(self, img, sigma, assume_blur):
        """_summary_
            get base image with 2x larger than raw image
        Args:
            img (2D array): Input image
            sigma (float, optional): _description_. Defaults to 1.6.
            assume_blur (float, optional): _description_. Defaults to 0.5.

        Returns:
            img array: Upsampling input image with gaussian
        """
        img = cv2.pyrUp(img)
        modify_sigma = (sigma**2 - (assume_blur * 2)**2)**0.5
        baseImgGaussian = cv2.GaussianBlur(
            img, (0, 0), sigmaX=modify_sigma, sigmaY=modify_sigma)
        return baseImgGaussian

    def ComputeOctaveNum(self, img_shape):
        """
            calculate octave numbers adaptively
            In this project we se octave as 4 directly
        Args:
            img_shape (h, w): image shape

        Returns:
            int: numbers of octave
        """
        # make sure the smallest side of pixel bigger than 3
        return int(round(np.log(min(img_shape)) / np.log(2) - 1))

    def GaussianKernelGenerate(self, num_interval, sigma):
        """_summary_
        generate different sigma gaussian kernels

        Args:
            num_interval (int): per octave interval + 3 image
            sigma (float): gaussian intensity

        Returns:
            list: list of gaussian_kernels intensity
        """
        image_num_in_octave = num_interval + 3
        k = 2**(1 / num_interval)
        gaussian_kernels = np.zeros(image_num_in_octave)
        gaussian_kernels[0] = sigma
        for idx in range(1, image_num_in_octave):
            previous_sigma = (k**(idx - 1)) * sigma
            current_sigma = k * previous_sigma
            gaussian_kernels[idx] = (current_sigma**2 - previous_sigma**2)**0.5
        return gaussian_kernels

    def GaussianImgGenerate(self, img, num_octaves, gaussian_kernels):
        """_summary_
            generate gaussian image in different octave and corresponding gaussian kernels
        Args:
            img (array): baseImg
            num_octaves (int): use octave number
            gaussian_kernels (list): gaussian kernel intensity

        Returns:
            array: list of gaussian image
        """
        list_gaussian_img = []
        for idx in range(num_octaves):
            per_octave_gaussian_image = []
            per_octave_gaussian_image.append(img)
            for kernel in gaussian_kernels[1:]:
                img = cv2.GaussianBlur(
                    img,
                    (0, 0),
                    sigmaX=kernel,
                    sigmaY=kernel)
                per_octave_gaussian_image.append(img)
            list_gaussian_img.append(per_octave_gaussian_image)
            base_octave_img = per_octave_gaussian_image[-3]
            img = cv2.pyrDown(base_octave_img)
        return np.array(list_gaussian_img, dtype=object)

    def DogImg(self, list_gaussian_img):
        """_summary_
            get difference between two gaussian image
        Args:
            list_gaussian_img (list): list of gaussian image

        Returns:
            array: list of Dog
        """
        list_dog = []
        for per_octave_gaussian_image in list_gaussian_img:
            list_octave_dog = []
            for img_1, img_2 in zip(
                    per_octave_gaussian_image,
                    per_octave_gaussian_image[1:]):
                list_octave_dog.append(img_2 - img_1)
            list_dog.append(list_octave_dog)
        return np.array(list_dog, dtype=object)

    def checkPixelExtreme(self, img1_zone, img2_zone, img3_zone, extreme_threshold):
        """_summary_
            check extreme value at central of cubic matrix
        Args:
            img1_zone (3x3 array): Compare image 9px
            img2_zone (3x3 array): Compare image with center point 8px
            img3_zone (3x3 array): Compare image 9px
            extreme_threshold (float)

        Returns:
            bool: whether pixel is extemum vlaues
        """
        # Check upper layer and lower layer and neighbor pixel with ceneter
        val_central_pixel = img2_zone[1, 1]
        if np.abs(val_central_pixel) > extreme_threshold:
            if val_central_pixel > 0:
                return np.all(val_central_pixel >= img1_zone) and \
                    np.all(val_central_pixel >= img3_zone) and \
                    np.all(val_central_pixel >= img2_zone[0, :]) and \
                    np.all(val_central_pixel >= img2_zone[2, :]) and \
                    val_central_pixel >= img2_zone[1, 2] and \
                    val_central_pixel >= img2_zone[1, 0]
            elif val_central_pixel < 0:
                return np.all(val_central_pixel <= img1_zone) and \
                    np.all(val_central_pixel <= img3_zone) and \
                    np.all(val_central_pixel <= img2_zone[0, :]) and \
                    np.all(val_central_pixel <= img2_zone[2, :]) and \
                    val_central_pixel <= img2_zone[1, 0] and \
                    val_central_pixel <= img2_zone[1, 2]
        return False

    def computeGradientAtCentral(self, pixel_cubic_mtx):
        """_summary_
            compute central gradient value
        Args:
            pixel_cubic_mtx (_type_): 3x3x3 cubic matrix

        Returns:
            array([dx, dy, dz])
        """
        dx = (pixel_cubic_mtx[1, 1, 2] - pixel_cubic_mtx[1, 1, 0]) / 2.
        dy = (pixel_cubic_mtx[1, 2, 1] - pixel_cubic_mtx[1, 0, 1]) / 2.
        dz = (pixel_cubic_mtx[2, 1, 1] - pixel_cubic_mtx[0, 1, 1]) / 2.
        return np.array([dx, dy, dz])

    def computeHessianAtCentral(self, pixel_cubic_mtx):
        """_summary_
            compute central hessian matrix
        Args:
            pixel_cubic_mtx (array): 3x3x3 cubic matrix

        Returns:
            hessian mtrix: 3x3 hessian array
        """
        val_central = pixel_cubic_mtx[1, 1, 1]
        dxx = pixel_cubic_mtx[1, 1, 2] - 2 * val_central + pixel_cubic_mtx[1, 1, 0]
        dxy = (
            pixel_cubic_mtx[1, 2, 2] - pixel_cubic_mtx[1, 2, 0] -
            pixel_cubic_mtx[1, 0, 2] + pixel_cubic_mtx[1, 0, 0]) / 4.
        dxz = (
            pixel_cubic_mtx[2, 1, 2] - pixel_cubic_mtx[2, 1, 0] -
            pixel_cubic_mtx[0, 1, 2] + pixel_cubic_mtx[0, 1, 0]) / 4.
        dyy = pixel_cubic_mtx[1, 2, 1] - 2 * val_central + pixel_cubic_mtx[1, 0, 1]
        dyz = (
            pixel_cubic_mtx[2, 2, 1] - pixel_cubic_mtx[2, 0, 1] -
            pixel_cubic_mtx[0, 2, 1] + pixel_cubic_mtx[0, 0, 1]) / 4.
        dzz = pixel_cubic_mtx[2, 1, 1] - 2 * val_central + pixel_cubic_mtx[0, 1, 1]
        return np.array([
            [dxx, dxy, dxz],
            [dxy, dyy, dyz],
            [dxz, dyz, dzz],
        ])

    def localizeExtremumViaQuadraticFit(
            self, i, j, image_index, octave_index, num_intervals,
            octave_dog_images, sigma, contrast_threshold,
            image_border_width, eigenvalue_ratio=10,
            num_attempts_until_convergence=5):
        """_summary_
            Use quadratic function predict precisely extremum value iteratively
        Args:
            i (int): image current x index
            j (int): image current y index
            image_index (int): search extreme pts img idx
            octave_index (int): current octave index to assign kp
            num_intervals (int): search pixel range intervals
            octave_dog_images (list): list dog image in current octave
            sigma (float): first assigned sigma value 1.6
            contrast_threshold (float): compare contrast threshold
            image_border_width (int): set up image border width to prevent border effect
            eigenvalue_ratio (int, optional): in paper is r recommended value is 10. Defaults to 10.
            num_attempts_until_convergence (int, optional): iterative find extreme times. Defaults to 5.

        Returns:
            keypoint, location img idx in octave
        """
        extreme_in_outside = False
        img_shape = octave_dog_images[0].shape
        for _idx in range(num_attempts_until_convergence):
            img1, img2, img3 = octave_dog_images[image_index - 1:image_index + 2]
            pixel_cubic_mtx = np.stack([
                img1[i - 1:i + 2, j - 1:j + 2],
                img2[i - 1:i + 2, j - 1:j + 2],
                img3[i - 1:i + 2, j - 1:j + 2],
            ]).astype('float32') / 255.
            gradient = self.computeGradientAtCentral(pixel_cubic_mtx)
            hessian = self.computeHessianAtCentral(pixel_cubic_mtx)
            # find pixel extreme value location offset
            extreme_offset = -np.linalg.inv(hessian).dot(gradient)
            # if one extreme point over 0.5 will find next extreme point
            if np.abs(extreme_offset[0] < 0.5) and \
                    np.abs(extreme_offset[1] < 0.5) and \
                    np.abs(extreme_offset[2] < 0.5):
                break
            # will add to original keypoint (width and height need transpose)
            j += int(round(extreme_offset[0]))
            i += int(round(extreme_offset[1]))
            image_index += int(round(extreme_offset[2]))
            # make sure extreme value in our set up available region
            if i < image_border_width or \
                    i >= (img_shape[0] - image_border_width) or \
                    j < image_border_width or \
                    j >= (img_shape[1] - image_border_width) or \
                    image_index < 1 or \
                    image_index > num_intervals:
                extreme_in_outside = True
                break
        if extreme_in_outside:
            return None
        # paper section 4.1
        val_update_extreme = pixel_cubic_mtx[1, 1, 1] + 0.5 * np.dot(gradient, extreme_offset)
        # reject low contrast unstable
        if np.abs(val_update_extreme) >= contrast_threshold / num_intervals:
            hessian_xy = hessian[:2, :2]
            trace_hessian_xy = np.trace(hessian_xy)
            det_hessian_xy = np.linalg.det(hessian_xy)
            # in determinate is non negative
            if det_hessian_xy > 0 and \
                    eigenvalue_ratio * (trace_hessian_xy ** 2) < ((eigenvalue_ratio + 1) ** 2) * det_hessian_xy:
                keypoint = cv2.KeyPoint()
                keypoint.pt = ((j + extreme_offset[0]) * (2**octave_index), (i + extreme_offset[1]) * (2**octave_index))
                keypoint.octave = octave_index + image_index * (2**8)
                keypoint.size = \
                    sigma * (2**((image_index + extreme_offset[2]) / np.float32(num_intervals))) * \
                    (2**(octave_index + 1))
                keypoint.response = np.abs(val_update_extreme)
                return keypoint, image_index
        return None

    def computeKeypointsWithOrientations(
            self, keypoint, idx_octave, gaussian_image, radius_factor=3,
            num_bins=36, peak_ratio=0.8, scale_factor=1.5):
        """
        Compute key point orinetation information

        Args:
            keypoint (cv2.KeyPoint()): keypoint object with pt octave size response information
            idx_octave (int): octave index
            gaussian_image (array): 2D gaussian image array
            radius_factor (int, optional): _description_. Defaults to 3.
            num_bins (int, optional): 36 bins seperate 360 degree. Defaults to 36.
            peak_ratio (float, optional): allow peak ratio. Defaults to 0.8.
            scale_factor (float, optional): gaussian blue scale factor. Defaults to 1.5.

        Returns:
            keypoints_with_orient: new assign cv2.KeyPoint() add orient information
        """
        keypoints_with_orient = []
        img_shape = gaussian_image.shape
        scale = scale_factor * keypoint.size / np.float32(2**(idx_octave + 1))
        weight_factor = -0.5 / (scale ** 2)
        radius = int(round(radius_factor * scale))
        raw_hist = np.zeros(num_bins)
        smooth_hist = np.zeros(num_bins)
        for i in range(-radius, radius + 1):
            region_y = int(round(keypoint.pt[1] / np.float32(2 ** idx_octave))) + i
            if region_y > 0 and region_y < img_shape[0] - 1:
                for j in range(-radius, radius + 1):
                    region_x = int(round(keypoint.pt[0] / np.float32(2 ** idx_octave))) + j
                    if region_x > 0 and region_x < img_shape[1] - 1:
                        # gaussian weighted window
                        weight = np.exp(weight_factor * (i**2 + j**2))
                        dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                        dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]
                        gradient_magnitude = (dx**2 + dy**2)**0.5
                        gradient_orient = np.rad2deg(np.arctan2(dy, dx))
                        idx_hist = int(round(gradient_orient * num_bins / 360.))
                        raw_hist[idx_hist % num_bins] += weight * gradient_magnitude
        for n in range(num_bins):
            # use 1 4 6 4 1 pascal triangle get gaussian weighted kernel
            smooth_hist[n] = (
                6 * raw_hist[n] +
                4 * (raw_hist[n - 1] + raw_hist[(n + 1) % num_bins]) +
                raw_hist[n - 2] + raw_hist[(n + 2) % num_bins]
            ) / 16.
        max_orient = np.max(smooth_hist)
        orient_peaks = np.where((smooth_hist > np.roll(smooth_hist, 1)) & (smooth_hist > np.roll(smooth_hist, -1)))[0]
        for idx_peak in orient_peaks:
            val_peak = smooth_hist[idx_peak]
            if val_peak >= peak_ratio * max_orient:
                val_left = smooth_hist[(idx_peak - 1) % num_bins]
                val_right = smooth_hist[(idx_peak + 1) % num_bins]
                interpolate_peak_idx = \
                    (idx_peak + 0.5 * (val_left - val_right) / (val_left - 2 * val_peak + val_right)) % num_bins
                orient = 360 - interpolate_peak_idx * 360 / num_bins
                new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, orient, keypoint.response, keypoint.octave)
                keypoints_with_orient.append(new_keypoint)
        return keypoints_with_orient

    def findExtremeScaleSpace(
            self, gaussian_images, dog_images, num_intervals,
            sigma, image_border_width, contrast_threshold=0.04):
        """
        Check exteme value and make sure extreme value in center pixel location

        Args:
            gaussian_images (list): list of gaussian images
            dog_images (list): list of differential of gaussian image
            num_intervals (int): intervals in per octave
            sigma (float): gaussian blur sigma size
            image_border_width (int): add image border prevent error
            contrast_threshold (float, optional): recommend contrast threshold. Defaults to 0.04.

        Returns:
            keypoint: cv2.keypoint object
        """
        threshold = np.floor(0.5 * contrast_threshold / num_intervals * 255)
        keypoints = []
        for idx_octave, octave_dog_img in enumerate(dog_images):
            for idx_img, (img1, img2, img3) in enumerate(zip(octave_dog_img, octave_dog_img[1:], octave_dog_img[2:])):
                for i in range(image_border_width, img1.shape[0] - image_border_width):
                    for j in range(image_border_width, img1.shape[1] - image_border_width):
                        if self.checkPixelExtreme(
                                img1[i - 1:i + 2, j - 1:j + 2],
                                img2[i - 1:i + 2, j - 1:j + 2],
                                img3[i - 1:i + 2, j - 1:j + 2],
                                threshold):
                            # check extreme value location
                            localization_result = self.localizeExtremumViaQuadraticFit(
                                i, j, idx_img + 1, idx_octave,
                                num_intervals, octave_dog_img, sigma,
                                contrast_threshold, image_border_width)
                            if localization_result is not None:
                                keypoint, idx_localized_image = localization_result
                                # compute kp orientation
                                keypoints_with_orientation = self.computeKeypointsWithOrientations(
                                    keypoint, idx_octave, gaussian_images[idx_octave][idx_localized_image])
                                for keypoint_with_orientation in keypoints_with_orientation:
                                    keypoints.append(keypoint_with_orientation)
        return keypoints

    def compareKeyPoints(self, kp1, kp2):
        """compare two keypoint for remove duplicated

        Args:
            kp1 (cv2.keypoint()): compare kp1
            kp2 (cv2.keypoint()): compare kp2

        Returns:
            float: compare result neg or pos
        """
        if kp1.pt[0] != kp2.pt[0]:
            return kp1.pt[0] - kp2.pt[0]
        if kp1.pt[1] != kp2.pt[1]:
            return kp1.pt[1] - kp2.pt[1]
        if kp1.size != kp2.size:
            return kp2.size - kp1.size
        if kp1.angle != kp2.angle:
            return kp1.angle - kp2.angle
        if kp1.response != kp2.response:
            return kp2.response - kp1.response
        if kp1.octave != kp2.octave:
            return kp1.octave - kp2.octave
        return kp2.class_id - kp1.class_id

    def removeDuplicatedKeyPoint(self, keypts):
        """ To remove duplicated points
            sort all keypoints and compare keypoints
        Args:
            keypts (cv2.keypoint()): list of keypoint

        Returns:
            unique keypoint: list of unique keypoint
        """
        if len(keypts) < 2:
            return keypts
        # To sort keypts in pt[0] pt[1] sizr angle response octave
        keypts.sort(key=functools.cmp_to_key(self.compareKeyPoints))
        unique_keypts = [keypts[0]]
        for next_kp in keypts[1:]:
            prev_keypts = unique_keypts[-1]
            if prev_keypts.pt[0] != next_kp.pt[0] or \
                    prev_keypts.pt[1] != next_kp.pt[1] or \
                    prev_keypts.size != next_kp.size or \
                    prev_keypts.angle != next_kp.angle:
                unique_keypts.append(next_kp)
        return unique_keypts

    def convertKeypoint2ImageSize(self, keypts):
        """_summary_
        resize ratio 2 -> have to multiply 0.5
        Args:
            keypts (list): list of cv2.Keypoint()

        Returns:
            list keypoints: restored keypoints
        """
        list_convert_keypts = []
        for keypoint in keypts:
            keypoint.pt = tuple(0.5 * np.array(keypoint.pt))
            keypoint.size *= 0.5
            # octave base on opencv sift octave value
            keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
            list_convert_keypts.append(keypoint)
        return list_convert_keypts

    def unpackOctave(self, kpts):
        """_summary_
            Due to we store octave and image index information in our octave parameters
            octave in 1-8 bit and image index in 9-16
        Args:
            kpts (cv2.Keypoint): need unpack keypoint octave information

        Returns:
            octave, layer, scale: description of current key point location
        """
        num_octave = kpts.octave & 255
        layer = (kpts.octave >> 8) & 255
        # solve negative problem e.g. scale 2 case
        if num_octave >= 128:
            num_octave = num_octave | -128
        scale = 1 / np.float32(1 << num_octave) if num_octave >= 0 else np.float(1 << -num_octave)
        return num_octave, layer, scale

    def trilinear_interpolation(self, magnitude, frac_row, frac_col, frac_orient):
        """_summary_
            trilinear interpolation to get more accurate magnitude
        Args:
            magnitude (float): histogram magnitude
            frac_row (float): real row -  floor real row
            frac_col (float): real col -  floor real col
            frac_orient (_type_): real orient -  floor real orient

        Returns:
            float: c000 to c111 magnitude
        """
        c0 = (1 - frac_row) * magnitude
        c1 = frac_row * magnitude
        c00 = (1 - frac_col) * c0
        c01 = frac_col * c0
        c10 = (1 - frac_col) * c1
        c11 = frac_col * c1
        c000 = c00 * (1 - frac_orient)
        c001 = c00 * frac_orient
        c010 = c01 * (1 - frac_orient)
        c011 = c01 * frac_orient
        c100 = c10 * (1 - frac_orient)
        c101 = c10 * frac_orient
        c110 = c11 * (1 - frac_orient)
        c111 = c11 * frac_orient
        return c000, c001, c010, c011, c100, c101, c110, c111

    def generateDescriptors(
            self, keypoints, gaussian_images, window_width=4,
            num_bins=8, scale_multiplier=3, val_descriptor_max=0.2):
        """To generate desciptors 4x4x8
            unpack octave to find keypoint location
            use location to modify select width
            and use trilinear interpolation to get point descriptors

        Args:
            keypoints (list): list of we detect keypoints
            gaussian_images (list): list of gaussaian images in per octave
            window_width (int, optional): 4X4 . Defaults to 4.
            num_bins (int, optional): 8 bins 45 degree. Defaults to 8.
            scale_multiplier (int, optional): we use paper recommend values 3. Defaults to 3.
            val_descriptor_max (float, optional): set maximum of descriptor also paper said. Defaults to 0.2.

        Returns:
            descriptors(array): list of descriptors for keypoints
        """
        list_descriptors = []

        for kp in keypoints:
            octave, layer, scale = self.unpackOctave(kp)
            # due to the 0 octave index is Upsampling image
            current_gaussian_image = gaussian_images[octave + 1, layer]
            pointlocation = np.round(scale * np.array(kp.pt)).astype('int')
            rows, cols = current_gaussian_image.shape
            bin_per_deg = num_bins / 360.
            angle = 360. - kp.angle
            rad = np.deg2rad(angle)
            weight_multiplier = -0.5 / ((window_width * 0.5)**2)
            cos_angle = np.cos(rad)
            sin_angle = np.sin(rad)
            # to prevent border effect we add 2 dimension and will remove head and tail latter
            hist_tensor = np.zeros((window_width + 2, window_width + 2, num_bins))
            list_rows = []
            list_cols = []
            list_magnitude = []
            list_orientation = []

            hist_width = scale_multiplier * 0.5 * scale * kp.size
            half_width = int(np.round(hist_width * (2**0.5) * (window_width + 1) * 0.5))
            # constrained width not over than current gaussian image size
            half_width = int(min(half_width, np.sqrt(rows**2 + cols**2)))

            for i in range(-half_width, half_width + 1):
                for j in range(-half_width, half_width + 1):
                    # use rotation matrix to make sure descriptor compare with same angle
                    rot_row = i * cos_angle + j * sin_angle
                    rot_col = i * (-sin_angle) + j * cos_angle
                    bin_row = (rot_row / hist_width) + 0.5 * window_width - 0.5
                    bin_col = (rot_col / hist_width) + 0.5 * window_width - 0.5

                    if bin_row > -1 and window_width > bin_row and bin_col > -1 and window_width > bin_col:
                        window_row = int(round(pointlocation[1] + i))
                        window_col = int(round(pointlocation[0] + j))
                        if window_row > 0 and rows - 1 > window_row and window_col > 0 and cols - 1 > window_col:
                            dx = current_gaussian_image[window_row, window_col + 1] - \
                                current_gaussian_image[window_row, window_col - 1]
                            dy = current_gaussian_image[window_row - 1, window_col] - \
                                current_gaussian_image[window_row + 1, window_col]
                            grad_magnitude = (dx * dx + dy * dy)**0.5
                            grad_orient = np.rad2deg(np.arctan2(dy, dx)) % 360
                            weight = np.exp(
                                weight_multiplier * ((rot_row / hist_width) ** 2 + (rot_col / hist_width) ** 2)
                            )
                            list_rows.append(bin_row)
                            list_cols.append(bin_col)
                            list_magnitude.append(weight * grad_magnitude)
                            list_orientation.append((grad_orient - angle) * bin_per_deg)
            for bin_row, bin_col, magnitude, orient in zip(list_rows, list_cols, list_magnitude, list_orientation):
                floor_bin_row, floor_bin_col, floor_bin_orient = np.floor([bin_row, bin_col, orient]).astype("int")
                frac_row = bin_row - floor_bin_row
                frac_col = bin_col - floor_bin_col
                frac_orient = orient - floor_bin_orient
                if floor_bin_orient >= num_bins:
                    floor_bin_orient -= num_bins
                elif floor_bin_orient < 0:
                    floor_bin_orient += num_bins

                c000, c001, c010, c011, c100, c101, c110, c111 = \
                    self.trilinear_interpolation(magnitude, frac_row, frac_col, frac_orient)

                hist_tensor[floor_bin_row + 1, floor_bin_col + 1, floor_bin_orient] += c000
                hist_tensor[floor_bin_row + 1, floor_bin_col + 1, (floor_bin_orient + 1) % num_bins] += c001
                hist_tensor[floor_bin_row + 1, floor_bin_col + 2, floor_bin_orient] += c010
                hist_tensor[floor_bin_row + 1, floor_bin_col + 2, (floor_bin_orient + 1) % num_bins] += c011
                hist_tensor[floor_bin_row + 2, floor_bin_col + 1, floor_bin_orient] += c100
                hist_tensor[floor_bin_row + 2, floor_bin_col + 1, (floor_bin_orient + 1) % num_bins] += c101
                hist_tensor[floor_bin_row + 2, floor_bin_col + 2, floor_bin_orient] += c110
                hist_tensor[floor_bin_row + 2, floor_bin_col + 2, (floor_bin_orient + 1) % num_bins] += c111
            # we selected our need 4X4 region from 6X6 window prevent edge impact
            vec_descriptor = hist_tensor[1:-1, 1:-1, :].flatten()
            # constrain max value
            threshold = np.linalg.norm(vec_descriptor) * val_descriptor_max
            vec_descriptor[vec_descriptor > threshold] = threshold
            vec_descriptor = vec_descriptor / max(np.linalg.norm(vec_descriptor), 1e-7)
            # over than 0.5 descriptor will become 255
            vec_descriptor = np.round(512 * vec_descriptor)
            vec_descriptor = np.clip(vec_descriptor, 0, 255)
            list_descriptors.append(vec_descriptor)
        return np.array(list_descriptors, dtype="float32")

    def SIFTDetectCompute(self):
        raw_image = self.raw_image.astype('float32')
        generate_base_start = time.time()
        baseImgGaussian = self.GetBaseImg(raw_image, self.sigma, self.assume_blur)
        generate_multi_gaussian_start = time.time()
        print("Generate Base Image Use: %s seconds" % (generate_multi_gaussian_start - generate_base_start))
        # num_octaves = self.ComputeOctaveNum(baseImgGaussian.shape)
        # In current project we set num_octave as 4
        num_octaves = 4
        gaussian_kernels = self.GaussianKernelGenerate(self.num_intervals, self.sigma)
        gaussian_images = self.GaussianImgGenerate(
            baseImgGaussian, num_octaves, gaussian_kernels)
        calculate_dog_start = time.time()
        print("Generate Multiple Gaussian Image Use: %s seconds" %
              (calculate_dog_start - generate_multi_gaussian_start))
        dog_images = self.DogImg(gaussian_images)
        keypoints = self.findExtremeScaleSpace(
            gaussian_images, dog_images, self.num_intervals,
            self.sigma, self.image_border_width, self.contrast_threshold
        )
        remove_conver_kp_start = time.time()
        print("Calculate Gaussian Image Use: %s seconds" % (remove_conver_kp_start - calculate_dog_start))
        keypoints = self.removeDuplicatedKeyPoint(keypoints)
        keypoints = self.convertKeypoint2ImageSize(keypoints)
        # descriptors = generateDescriptors(keypoints, raw_image)
        generate_descriptors_start = time.time()
        print("Remove and convert kp to original size Use: %s seconds" %
              (generate_descriptors_start - remove_conver_kp_start))
        descriptors = self.generateDescriptors(keypoints, gaussian_images)
        print("Generate Descriptors Use: %s seconds" % (time.time() - generate_descriptors_start))
        return keypoints, descriptors
