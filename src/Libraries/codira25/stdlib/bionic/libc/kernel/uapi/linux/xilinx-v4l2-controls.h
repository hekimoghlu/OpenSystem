/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 29, 2024.
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
#ifndef __UAPI_XILINX_V4L2_CONTROLS_H__
#define __UAPI_XILINX_V4L2_CONTROLS_H__
#include <linux/v4l2-controls.h>
#define V4L2_CID_XILINX_OFFSET 0xc000
#define V4L2_CID_XILINX_BASE (V4L2_CID_USER_BASE + V4L2_CID_XILINX_OFFSET)
#define V4L2_CID_XILINX_TPG (V4L2_CID_USER_BASE + 0xc000)
#define V4L2_CID_XILINX_TPG_CROSS_HAIRS (V4L2_CID_XILINX_TPG + 1)
#define V4L2_CID_XILINX_TPG_MOVING_BOX (V4L2_CID_XILINX_TPG + 2)
#define V4L2_CID_XILINX_TPG_COLOR_MASK (V4L2_CID_XILINX_TPG + 3)
#define V4L2_CID_XILINX_TPG_STUCK_PIXEL (V4L2_CID_XILINX_TPG + 4)
#define V4L2_CID_XILINX_TPG_NOISE (V4L2_CID_XILINX_TPG + 5)
#define V4L2_CID_XILINX_TPG_MOTION (V4L2_CID_XILINX_TPG + 6)
#define V4L2_CID_XILINX_TPG_MOTION_SPEED (V4L2_CID_XILINX_TPG + 7)
#define V4L2_CID_XILINX_TPG_CROSS_HAIR_ROW (V4L2_CID_XILINX_TPG + 8)
#define V4L2_CID_XILINX_TPG_CROSS_HAIR_COLUMN (V4L2_CID_XILINX_TPG + 9)
#define V4L2_CID_XILINX_TPG_ZPLATE_HOR_START (V4L2_CID_XILINX_TPG + 10)
#define V4L2_CID_XILINX_TPG_ZPLATE_HOR_SPEED (V4L2_CID_XILINX_TPG + 11)
#define V4L2_CID_XILINX_TPG_ZPLATE_VER_START (V4L2_CID_XILINX_TPG + 12)
#define V4L2_CID_XILINX_TPG_ZPLATE_VER_SPEED (V4L2_CID_XILINX_TPG + 13)
#define V4L2_CID_XILINX_TPG_BOX_SIZE (V4L2_CID_XILINX_TPG + 14)
#define V4L2_CID_XILINX_TPG_BOX_COLOR (V4L2_CID_XILINX_TPG + 15)
#define V4L2_CID_XILINX_TPG_STUCK_PIXEL_THRESH (V4L2_CID_XILINX_TPG + 16)
#define V4L2_CID_XILINX_TPG_NOISE_GAIN (V4L2_CID_XILINX_TPG + 17)
#endif
