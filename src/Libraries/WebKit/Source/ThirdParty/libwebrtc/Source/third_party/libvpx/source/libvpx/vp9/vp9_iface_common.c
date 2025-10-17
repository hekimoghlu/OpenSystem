/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 14, 2021.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#include "vp9/vp9_iface_common.h"
void yuvconfig2image(vpx_image_t *img, const YV12_BUFFER_CONFIG *yv12,
                     void *user_priv) {
  /** vpx_img_wrap() doesn't allow specifying independent strides for
   * the Y, U, and V planes, nor other alignment adjustments that
   * might be representable by a YV12_BUFFER_CONFIG, so we just
   * initialize all the fields.*/
  int bps;
  if (!yv12->subsampling_y) {
    if (!yv12->subsampling_x) {
      img->fmt = VPX_IMG_FMT_I444;
      bps = 24;
    } else {
      img->fmt = VPX_IMG_FMT_I422;
      bps = 16;
    }
  } else {
    if (!yv12->subsampling_x) {
      img->fmt = VPX_IMG_FMT_I440;
      bps = 16;
    } else {
      img->fmt = VPX_IMG_FMT_I420;
      bps = 12;
    }
  }
  img->cs = yv12->color_space;
  img->range = yv12->color_range;
  img->bit_depth = 8;
  img->w = yv12->y_stride;
  img->h = ALIGN_POWER_OF_TWO(yv12->y_height + 2 * VP9_ENC_BORDER_IN_PIXELS, 3);
  img->d_w = yv12->y_crop_width;
  img->d_h = yv12->y_crop_height;
  img->r_w = yv12->render_width;
  img->r_h = yv12->render_height;
  img->x_chroma_shift = yv12->subsampling_x;
  img->y_chroma_shift = yv12->subsampling_y;
  img->planes[VPX_PLANE_Y] = yv12->y_buffer;
  img->planes[VPX_PLANE_U] = yv12->u_buffer;
  img->planes[VPX_PLANE_V] = yv12->v_buffer;
  img->planes[VPX_PLANE_ALPHA] = NULL;
  img->stride[VPX_PLANE_Y] = yv12->y_stride;
  img->stride[VPX_PLANE_U] = yv12->uv_stride;
  img->stride[VPX_PLANE_V] = yv12->uv_stride;
  img->stride[VPX_PLANE_ALPHA] = yv12->y_stride;
#if CONFIG_VP9_HIGHBITDEPTH
  if (yv12->flags & YV12_FLAG_HIGHBITDEPTH) {
    // vpx_image_t uses byte strides and a pointer to the first byte
    // of the image.
    img->fmt = (vpx_img_fmt_t)(img->fmt | VPX_IMG_FMT_HIGHBITDEPTH);
    img->bit_depth = yv12->bit_depth;
    img->planes[VPX_PLANE_Y] = (uint8_t *)CONVERT_TO_SHORTPTR(yv12->y_buffer);
    img->planes[VPX_PLANE_U] = (uint8_t *)CONVERT_TO_SHORTPTR(yv12->u_buffer);
    img->planes[VPX_PLANE_V] = (uint8_t *)CONVERT_TO_SHORTPTR(yv12->v_buffer);
    img->planes[VPX_PLANE_ALPHA] = NULL;
    img->stride[VPX_PLANE_Y] = 2 * yv12->y_stride;
    img->stride[VPX_PLANE_U] = 2 * yv12->uv_stride;
    img->stride[VPX_PLANE_V] = 2 * yv12->uv_stride;
    img->stride[VPX_PLANE_ALPHA] = 2 * yv12->y_stride;
  }
#endif  // CONFIG_VP9_HIGHBITDEPTH
  img->bps = bps;
  img->user_priv = user_priv;
  img->img_data = yv12->buffer_alloc;
  img->img_data_owner = 0;
  img->self_allocd = 0;
}

vpx_codec_err_t image2yuvconfig(const vpx_image_t *img,
                                YV12_BUFFER_CONFIG *yv12) {
  yv12->y_buffer = img->planes[VPX_PLANE_Y];
  yv12->u_buffer = img->planes[VPX_PLANE_U];
  yv12->v_buffer = img->planes[VPX_PLANE_V];

  yv12->y_crop_width = img->d_w;
  yv12->y_crop_height = img->d_h;
  yv12->render_width = img->r_w;
  yv12->render_height = img->r_h;
  yv12->y_width = img->d_w;
  yv12->y_height = img->d_h;

  yv12->uv_width = img->x_chroma_shift == 1 || img->fmt == VPX_IMG_FMT_NV12
                       ? (1 + yv12->y_width) / 2
                       : yv12->y_width;
  yv12->uv_height =
      img->y_chroma_shift == 1 ? (1 + yv12->y_height) / 2 : yv12->y_height;
  yv12->uv_crop_width = yv12->uv_width;
  yv12->uv_crop_height = yv12->uv_height;

  yv12->y_stride = img->stride[VPX_PLANE_Y];
  yv12->uv_stride = img->stride[VPX_PLANE_U];
  yv12->color_space = img->cs;
  yv12->color_range = img->range;

#if CONFIG_VP9_HIGHBITDEPTH
  if (img->fmt & VPX_IMG_FMT_HIGHBITDEPTH) {
    // In vpx_image_t
    //     planes point to uint8 address of start of data
    //     stride counts uint8s to reach next row
    // In YV12_BUFFER_CONFIG
    //     y_buffer, u_buffer, v_buffer point to uint16 address of data
    //     stride and border counts in uint16s
    // This means that all the address calculations in the main body of code
    // should work correctly.
    // However, before we do any pixel operations we need to cast the address
    // to a uint16 ponter and double its value.
    yv12->y_buffer = CONVERT_TO_BYTEPTR(yv12->y_buffer);
    yv12->u_buffer = CONVERT_TO_BYTEPTR(yv12->u_buffer);
    yv12->v_buffer = CONVERT_TO_BYTEPTR(yv12->v_buffer);
    yv12->y_stride >>= 1;
    yv12->uv_stride >>= 1;
    yv12->flags = YV12_FLAG_HIGHBITDEPTH;
  } else {
    yv12->flags = 0;
  }
  yv12->border = (yv12->y_stride - img->w) / 2;
#else
  yv12->border = (img->stride[VPX_PLANE_Y] - img->w) / 2;
#endif  // CONFIG_VP9_HIGHBITDEPTH
  yv12->subsampling_x = img->x_chroma_shift;
  yv12->subsampling_y = img->y_chroma_shift;
  // When reading the data, UV are in one plane for NV12 format, thus
  // x_chroma_shift is 0. After converting, UV are in separate planes, and
  // subsampling_x should be set to 1.
  if (img->fmt == VPX_IMG_FMT_NV12) yv12->subsampling_x = 1;
  return VPX_CODEC_OK;
}
