/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 16, 2025.
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
#ifndef __UAPI_THP7312_H_
#define __UAPI_THP7312_H_
#include <linux/v4l2-controls.h>
#define V4L2_CID_THP7312_LOW_LIGHT_COMPENSATION (V4L2_CID_USER_THP7312_BASE + 0x01)
#define V4L2_CID_THP7312_AUTO_FOCUS_METHOD (V4L2_CID_USER_THP7312_BASE + 0x02)
#define V4L2_CID_THP7312_NOISE_REDUCTION_AUTO (V4L2_CID_USER_THP7312_BASE + 0x03)
#define V4L2_CID_THP7312_NOISE_REDUCTION_ABSOLUTE (V4L2_CID_USER_THP7312_BASE + 0x04)
#endif
