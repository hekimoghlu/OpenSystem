/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 27, 2025.
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
#ifndef __LINUX_IVTV_H__
#define __LINUX_IVTV_H__
#include <linux/compiler.h>
#include <linux/types.h>
#include <linux/videodev2.h>
struct ivtv_dma_frame {
  enum v4l2_buf_type type;
  __u32 pixelformat;
  void  * y_source;
  void  * uv_source;
  struct v4l2_rect src;
  struct v4l2_rect dst;
  __u32 src_width;
  __u32 src_height;
};
#define IVTV_IOC_DMA_FRAME _IOW('V', BASE_VIDIOC_PRIVATE + 0, struct ivtv_dma_frame)
#define IVTV_IOC_PASSTHROUGH_MODE _IOW('V', BASE_VIDIOC_PRIVATE + 1, int)
#define IVTV_SLICED_TYPE_TELETEXT_B V4L2_MPEG_VBI_IVTV_TELETEXT_B
#define IVTV_SLICED_TYPE_CAPTION_525 V4L2_MPEG_VBI_IVTV_CAPTION_525
#define IVTV_SLICED_TYPE_WSS_625 V4L2_MPEG_VBI_IVTV_WSS_625
#define IVTV_SLICED_TYPE_VPS V4L2_MPEG_VBI_IVTV_VPS
#endif
