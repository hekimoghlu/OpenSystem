/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 15, 2024.
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
#ifndef VPX_VPX_DSP_VARIANCE_H_
#define VPX_VPX_DSP_VARIANCE_H_

#include "./vpx_config.h"

#include "vpx/vpx_integer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FILTER_BITS 7
#define FILTER_WEIGHT 128

typedef unsigned int (*vpx_sad_fn_t)(const uint8_t *src_ptr, int src_stride,
                                     const uint8_t *ref_ptr, int ref_stride);

typedef unsigned int (*vpx_sad_avg_fn_t)(const uint8_t *src_ptr, int src_stride,
                                         const uint8_t *ref_ptr, int ref_stride,
                                         const uint8_t *second_pred);

typedef void (*vp8_copy32xn_fn_t)(const uint8_t *src_ptr, int src_stride,
                                  uint8_t *ref_ptr, int ref_stride, int n);

typedef void (*vpx_sad_multi_fn_t)(const uint8_t *src_ptr, int src_stride,
                                   const uint8_t *ref_ptr, int ref_stride,
                                   unsigned int *sad_array);

typedef void (*vpx_sad_multi_d_fn_t)(const uint8_t *src_ptr, int src_stride,
                                     const uint8_t *const b_array[],
                                     int ref_stride, unsigned int *sad_array);

typedef unsigned int (*vpx_variance_fn_t)(const uint8_t *src_ptr,
                                          int src_stride,
                                          const uint8_t *ref_ptr,
                                          int ref_stride, unsigned int *sse);

typedef unsigned int (*vpx_subpixvariance_fn_t)(
    const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset,
    const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

typedef unsigned int (*vpx_subp_avg_variance_fn_t)(
    const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset,
    const uint8_t *ref_ptr, int ref_stride, unsigned int *sse,
    const uint8_t *second_pred);

#if CONFIG_VP8
typedef struct variance_vtable {
  vpx_sad_fn_t sdf;
  vpx_variance_fn_t vf;
  vpx_subpixvariance_fn_t svf;
  vpx_sad_multi_d_fn_t sdx4df;
#if VPX_ARCH_X86 || VPX_ARCH_X86_64
  vp8_copy32xn_fn_t copymem;
#endif
} vp8_variance_fn_ptr_t;
#endif  // CONFIG_VP8

#if CONFIG_VP9
typedef struct vp9_variance_vtable {
  vpx_sad_fn_t sdf;
  // Same as normal sad, but downsample the rows by a factor of 2.
  vpx_sad_fn_t sdsf;
  vpx_sad_avg_fn_t sdaf;
  vpx_variance_fn_t vf;
  vpx_subpixvariance_fn_t svf;
  vpx_subp_avg_variance_fn_t svaf;
  vpx_sad_multi_d_fn_t sdx4df;
  // Same as sadx4, but downsample the rows by a factor of 2.
  vpx_sad_multi_d_fn_t sdsx4df;
} vp9_variance_fn_ptr_t;
#endif  // CONFIG_VP9

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPX_DSP_VARIANCE_H_
