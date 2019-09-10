# coding=utf-8
from .cuda_interface import Stream, load_kernel, get_grid_block
import numpy as np
import torch

__all__ = ['BoxClip', 'BoxIou', 'BoxLoc', 'BoxSelect', 'BoxUpdate', 'AnchorCreate', 'box_loc', 'box_iou', 'box_clip',
           'box_update', 'box_select', 'anchor_create', 'box_clip_filter']


cuda_code = """
extern "C" {

__global__ void box_clip(const float* box, const float* image_info, float* output, int b_size, int k_size) {
    for (int ind = blockIdx.x * blockDim.x + threadIdx.x; ind < b_size * k_size * 4; ind += gridDim.x * blockDim.x) {
        int o_ind = ind % 4;
        int b_ind = ind / (k_size * 4);
        float box_value = box[ind];
        int h_or_w = o_ind % 2 ? 0 : 1;
        float info_value = image_info[b_ind*2 + h_or_w];
        output[ind] = max(0., min(box_value, info_value));
    }
}

__global__ void box_update(const float* box, const float* delta, float* output, int count) {
    for (int ind = blockIdx.x * blockDim.x + threadIdx.x; ind < count; ind += gridDim.x * blockDim.x) {
        int start_ind = ind * 4;
        float b_w0 = box[start_ind];
        float b_h0 = box[start_ind + 1];
        float b_w1 = box[start_ind + 2];
        float b_h1 = box[start_ind + 3];
        float b_w = b_w1 - b_w0;
        float b_h = b_h1 - b_h0;
        float bw_c = b_w0 + b_w * 0.5f;
        float bh_c = b_h0 + b_h * 0.5f;

        float dw_c = delta[start_ind];
        float dh_c = delta[start_ind + 1];
        float dw = delta[start_ind + 2];
        float dh = delta[start_ind + 3];

        float pred_h_c = dh_c * b_h + bh_c;
        float pred_w_c = dw_c * b_w + bw_c;
        float pred_h = exp(dh) * b_h;
        float pred_w = exp(dw) * b_w;

        output[start_ind] = pred_w_c - 0.5f * pred_w;
        output[start_ind+1] = pred_h_c - 0.5f * pred_h;
        output[start_ind+2] = pred_w_c + 0.5f * pred_w;
        output[start_ind+3] = pred_h_c + 0.5f * pred_h;
    }
}


__global__ void anchor_create(const float* base_anchor, float* anchor, int feat_stride, int height, int width, int c_num) {
    for (int ind = blockIdx.x * blockDim.x + threadIdx.x; ind < height*width*c_num*4; ind += gridDim.x * blockDim.x) {
        int o_ind = ind % 4;
        int c_ind = ind % (c_num * 4) / 4;
        int w_ind = ind / (c_num * 4) % width;
        int h_ind = ind / (width * c_num * 4);
        float shift_value = (o_ind % 2 == 0) ? (w_ind + 0.5) * feat_stride : (h_ind + 0.5) * feat_stride;
        float base_anchor_value = base_anchor[c_ind*4 + o_ind];
        anchor[ind] = shift_value + base_anchor_value;
    }
}


__global__ void box_select(const float* box, unsigned char* output, const float box_h, const float box_w,
                           const float allowed_boarder, const int count) {
    for (int thr_idx = blockIdx.x * blockDim.x + threadIdx.x; thr_idx < count; thr_idx += gridDim.x * blockDim.x) {
        int cur_box_idx = thr_idx * 4;
        float box_w0 = box[cur_box_idx], box_h0 = box[cur_box_idx + 1], box_w1 = box[cur_box_idx + 2], box_h1 = box[cur_box_idx + 3];
        output[thr_idx] = (box_w0 >= -allowed_boarder) & (box_h0 >= -allowed_boarder)
                          & (box_w1 <= box_w + allowed_boarder) & (box_h1 <= box_h + allowed_boarder);
    }
}


__device__ inline float iou(float const *const a, float const *const b) {
    float Sa = (a[2] - a[0]) * (a[3] - a[1]);
    float Sb = (b[2] - b[0]) * (b[3] - b[1]);
    if (Sa == 0.f || Sb == 0.f ) {
        return -1;
    }
    float left = max(a[0], b[0]), right = min(a[2], b[2]);
    float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
    float width = max(right - left, 0.f), height = max(bottom - top, 0.f);
    float interS = width * height;
    return interS / (Sa + Sb - interS);
}



__global__ void box_iou(const float* box_a, const float* box_b, float* output, const int a_k_size,
                        const int b_k_size, const int count) {
    for (int thr_idx = blockIdx.x * blockDim.x + threadIdx.x; thr_idx < count; thr_idx += gridDim.x * blockDim.x) {
        int a_k_ind = thr_idx / b_k_size % a_k_size;
        int b_k_ind = thr_idx % b_k_size;
        int batch_id = thr_idx / b_k_size / a_k_size;
        
        int a_start_ind = (batch_id * a_k_size + a_k_ind) * 4;
        int b_start_ind = (batch_id * b_k_size + b_k_ind) * 4;
        output[thr_idx] = iou(box_a+a_start_ind, box_b+b_start_ind);
    }
}



__global__ void box_loc(const float* box, const float* gt_box, float* output, int count) {
    for (int thd_idx = blockIdx.x * blockDim.x + threadIdx.x; thd_idx < count; thd_idx += gridDim.x * blockDim.x) {
        int box_start_ind = thd_idx * 4;
        float b_w0 = box[box_start_ind];
        float b_h0 = box[box_start_ind + 1];
        float b_w1 = box[box_start_ind + 2];
        float b_h1 = box[box_start_ind + 3];
        float b_w = b_w1 - b_w0;
        float b_h = b_h1 - b_h0;
        float bw_c = b_w0 + b_w * 0.5f;
        float bh_c = b_h0 + b_h * 0.5f;

        float gt_w0 = gt_box[box_start_ind];
        float gt_h0 = gt_box[box_start_ind + 1];
        float gt_w1 = gt_box[box_start_ind + 2];
        float gt_h1 = gt_box[box_start_ind + 3];
        float gt_w = gt_w1 - gt_w0;
        float gt_h = gt_h1 - gt_h0;
        float gtw_c = gt_w0 + gt_w * 0.5f;
        float gth_c = gt_h0 + gt_h * 0.5f;

        output[box_start_ind] = (gtw_c - bw_c) / b_w;
        output[box_start_ind+1] = (gth_c - bh_c) / b_h;
        output[box_start_ind+2] = log(gt_w / b_w);
        output[box_start_ind+3] = log(gt_h / b_h);
    }
}


__global__ void box_clip_filter(const float* box, const float* image_info, float* output, unsigned char* mask,
                              const int k_size, const float min_size, const int thrd_num) {
    for (int ind = blockIdx.x * blockDim.x + threadIdx.x; ind < thrd_num; ind += gridDim.x * blockDim.x) {
        int b_ind = ind / k_size;
        int k_ind = ind % k_size;
        int box_ind = (b_ind * k_size + k_ind) * 4;
        float box_w0 = box[box_ind], box_h0 = box[box_ind+1], box_w1 = box[box_ind+2], box_h1 = box[box_ind+3];
        int image_info_ind = b_ind * 3;
        float image_h = image_info[image_info_ind], image_w = image_info[image_info_ind+1], image_scale = image_info[image_info_ind+2];
        float output_w0 = max(0., min(box_w0, image_w));
        float output_h0 = max(0., min(box_h0, image_h));
        float output_w1 = max(0., min(box_w1, image_w));
        float output_h1 = max(0., min(box_h1, image_h));
        output[box_ind] = output_w0;
        output[box_ind+1] = output_h0;
        output[box_ind+2] = output_w1;
        output[box_ind+3] = output_h1;
        float output_w = output_w1 - output_w0;
        float output_h = output_h1 - output_h0;
        mask[b_ind * k_size + k_ind] = (output_w >= min_size*image_scale) & (output_h >= image_scale);
    }
}


}

"""


class Base(object):
    def __init__(self):
        self.kernel_func = None

    def __call__(self, *args, **kwargs):
        return self.kernel_func(*args, **kwargs)


class BoxUpdate(Base):
    def __init__(self):
        super().__init__()
        self.kernel_func = load_kernel('box_update', cuda_code, self.kernel_wrap)

    @staticmethod
    def kernel_wrap(kernel_func):
        def wrap(box, delta):
            assert box.shape == delta.shape
            if not box.is_contiguous():
                box = box.contiguous()
            if not delta.is_contiguous():
                delta = delta.contiguous()
            output = box.clone()
            count = int(output.numel() / 4)
            args = (box.data_ptr(), delta.data_ptr(), output.data_ptr(), np.int32(count))
            grid, block = get_grid_block(count, threads_per_block=512)
            stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
            kernel_func(args=args, grid=grid, block=block, stream=stream)
            return output

        return wrap


class BoxClip(Base):
    def __init__(self):
        super().__init__()
        self.kernel_func = load_kernel('box_clip', cuda_code, self.kernel_wrap)

    @staticmethod
    def kernel_wrap(kernel_func):
        def wrap(box, im_info):
            assert box.dim() == 3 and im_info.dim() == 2
            if not box.is_contiguous():
                box = box.contiguous()
            if not im_info.is_contiguous():
                im_info = im_info.contiguous()
            output = box.clone()
            args = (
                box.data_ptr(), im_info.data_ptr(), output.data_ptr(), np.int32(box.size(0)), np.int32(box.size(1)))
            grid, block = get_grid_block(output.numel(), threads_per_block=512)
            stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
            kernel_func(args=args, grid=grid, block=block, stream=stream)
            return output

        return wrap


class AnchorCreate(Base):
    def __init__(self):
        super().__init__()
        self.kernel_func = load_kernel('anchor_create', cuda_code, self.kernel_wrap)

    @staticmethod
    def kernel_wrap(kernel_func):
        def wrap(base_anchor, feat_stride, height, width):
            if not base_anchor.is_contiguous():
                base_anchor = base_anchor.contiguous()
            anchor = base_anchor.new(height, width, base_anchor.size(0) * 4)
            args = (base_anchor.data_ptr(), anchor.data_ptr(), np.int32(feat_stride), np.int32(height),
                    np.int32(width), np.int32(base_anchor.size(0)))
            grid, block = get_grid_block(anchor.numel(), threads_per_block=512)
            stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
            kernel_func(args=args, grid=grid, block=block, stream=stream)
            return anchor
        return wrap


class BoxSelect(Base):
    def __init__(self):
        super().__init__()
        self.kernel_func = load_kernel('box_select', cuda_code, self.kernel_wrap)

    @staticmethod
    def kernel_wrap(kernel_func):
        def wrap(box, box_h, box_w, allowed_border):
            if not box.is_contiguous():
                box.contiguous()
            box_count = int(box.numel() / 4)
            output_mask = torch.ByteTensor(box_count).view(*box.shape[:-1]).cuda()
            args = (box.data_ptr(), output_mask.data_ptr(), np.float32(box_h), np.float32(box_w),
                    np.float32(allowed_border), np.int32(box_count))
            grid, block = get_grid_block(box_count, threads_per_block=512)
            stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
            kernel_func(args=args, grid=grid, block=block, stream=stream)
            return output_mask

        return wrap


class BoxIou(Base):
    def __init__(self):
        super().__init__()
        self.kernel_func = load_kernel('box_iou', cuda_code, self.kernel_wrap)

    @staticmethod
    def kernel_wrap(kernel_func):
        def wrap(box_a, box_b):
            assert box_a.size(-1) == box_b.size(-1) == 4 and box_a.dim() == box_b.dim() == 3 and box_a.size(
                0) == box_b.size(0)
            if not box_a.is_contiguous():
                box_a = box_a.contiguous()
            if not box_b.is_contiguous():
                box_b = box_b.contiguous()
            batch_size, a_k_size = box_a.shape[:2]
            b_k_size = box_b.size(1)
            output = torch.FloatTensor(batch_size, a_k_size, b_k_size).cuda()
            args = (box_a.data_ptr(), box_b.data_ptr(), output.data_ptr(), np.int32(a_k_size), np.int32(b_k_size),
                    np.int32(output.numel()))
            grid, block = get_grid_block(output.numel(), threads_per_block=512)
            stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
            kernel_func(args=args, grid=grid, block=block, stream=stream)
            return output

        return wrap


class BoxLoc(Base):
    def __init__(self):
        super().__init__()
        self.kernel_func = load_kernel('box_loc', cuda_code, self.kernel_wrap)

    @staticmethod
    def kernel_wrap(kernel_func):
        def wrap(box, gt):
            assert box.shape == gt.shape
            if not box.is_contiguous():
                box = box.contiguous()
            if not gt.is_contiguous():
                gt = gt.contiguous()
            output = box.clone()
            count = int(box.numel() / 4)
            args = (box.data_ptr(), gt.data_ptr(), output.data_ptr(), np.int32(count))
            grid, block = get_grid_block(count, threads_per_block=512)
            stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
            kernel_func(args=args, grid=grid, block=block, stream=stream)
            return output

        return wrap


def box_loc(box, gt):
    assert box.shape == gt.shape
    if not box.is_contiguous():
        box = box.contiguous()
    if not gt.is_contiguous():
        gt = gt.contiguous()
    output = box.clone()
    thread_num = int(box.numel() / 4)
    args = (box.data_ptr(), gt.data_ptr(), output.data_ptr(), np.int32(thread_num))
    grid, block = get_grid_block(thread_num, threads_per_block=512)
    stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
    kernel_func = load_kernel('box_loc', cuda_code)
    kernel_func(args=args, grid=grid, block=block, stream=stream)
    return output


def box_iou(box_a, box_b):
    assert box_a.size(-1) == box_b.size(-1) == 4 and box_a.dim() == box_b.dim() == 3 and box_a.size(0) == box_b.size(0)
    if not box_a.is_contiguous():
        box_a = box_a.contiguous()
    if not box_b.is_contiguous():
        box_b = box_b.contiguous()
    batch_size, a_k_size = box_a.shape[:2]
    b_k_size = box_b.size(1)
    output = torch.FloatTensor(batch_size, a_k_size, b_k_size).cuda()
    args = (box_a.data_ptr(), box_b.data_ptr(), output.data_ptr(), np.int32(a_k_size), np.int32(b_k_size),
            np.int32(output.numel()))
    grid, block = get_grid_block(output.numel(), threads_per_block=512)
    stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
    kernel_func = load_kernel('box_iou', cuda_code)
    kernel_func(args=args, grid=grid, block=block, stream=stream)
    return output


def box_select(box, box_h, box_w, allowed_border):
    if not box.is_contiguous():
        box.contiguous()
    box_count = int(box.numel() / 4)
    output_mask = torch.ByteTensor(box_count).view(*box.shape[:-1]).cuda()
    args = (box.data_ptr(), output_mask.data_ptr(), np.float32(box_h), np.float32(box_w),
            np.float32(allowed_border), np.int32(box_count))
    grid, block = get_grid_block(box_count, threads_per_block=512)
    stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
    kernel_func = load_kernel('box_select', cuda_code)
    kernel_func(args=args, grid=grid, block=block, stream=stream)
    return output_mask


def anchor_create(base_anchor, feat_stride, height, width):
    if not base_anchor.is_contiguous():
        base_anchor = base_anchor.contiguous()
    anchor = base_anchor.new(height, width, base_anchor.size(0) * 4)
    args = (base_anchor.data_ptr(), anchor.data_ptr(), np.int32(feat_stride), np.int32(height),
            np.int32(width), np.int32(base_anchor.size(0)))
    grid, block = get_grid_block(anchor.numel(), threads_per_block=512)
    stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
    kernel_func = load_kernel('anchor_create', cuda_code)
    kernel_func(args=args, grid=grid, block=block, stream=stream)
    return anchor


def box_clip(box, im_info):
    assert box.dim() == 3 and im_info.dim() == 2
    if not box.is_contiguous():
        box = box.contiguous()
    if not im_info.is_contiguous():
        im_info = im_info.contiguous()
    output = box.clone()
    args = (
        box.data_ptr(), im_info.data_ptr(), output.data_ptr(), np.int32(box.size(0)), np.int32(box.size(1)))
    grid, block = get_grid_block(output.numel(), threads_per_block=512)
    stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
    kernel_func = load_kernel('box_clip', cuda_code)
    kernel_func(args=args, grid=grid, block=block, stream=stream)
    return output


def box_update(box, delta):
    assert box.shape == delta.shape
    if not box.is_contiguous():
        box = box.contiguous()
    if not delta.is_contiguous():
        delta = delta.contiguous()
    output = box.clone()
    count = int(output.numel() / 4)
    args = (box.data_ptr(), delta.data_ptr(), output.data_ptr(), np.int32(count))
    grid, block = get_grid_block(count, threads_per_block=512)
    stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
    kernel_func = load_kernel('box_update', cuda_code)
    kernel_func(args=args, grid=grid, block=block, stream=stream)
    return output


def box_clip_filter(box, im_info, min_size):
    assert box.dim() == 3 and im_info.dim() == 2
    if not box.is_contiguous():
        box = box.contiguous()
    if not im_info.is_contiguous():
        im_info = im_info.contiguous()
    output = box.clone()
    thread_num = box.shape[0] * box.shape[1]
    mask = torch.cuda.ByteTensor(*box.shape[:2])
    args = (
        box.data_ptr(), im_info.data_ptr(), output.data_ptr(), mask.data_ptr(), np.int32(box.size(1)),
        np.float32(min_size), np.int32(thread_num))
    grid, block = get_grid_block(thread_num, threads_per_block=512)
    stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
    kernel_func = load_kernel('box_clip_filter', cuda_code)
    kernel_func(args=args, grid=grid, block=block, stream=stream)
    return output, mask
