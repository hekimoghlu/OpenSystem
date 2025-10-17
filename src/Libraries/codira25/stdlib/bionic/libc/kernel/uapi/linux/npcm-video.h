/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 2, 2025.
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
#ifndef _UAPI_LINUX_NPCM_VIDEO_H
#define _UAPI_LINUX_NPCM_VIDEO_H
#include <linux/v4l2-controls.h>
#define V4L2_CID_NPCM_CAPTURE_MODE (V4L2_CID_USER_NPCM_BASE + 0)
enum v4l2_npcm_capture_mode {
  V4L2_NPCM_CAPTURE_MODE_COMPLETE = 0,
  V4L2_NPCM_CAPTURE_MODE_DIFF = 1,
};
#define V4L2_CID_NPCM_RECT_COUNT (V4L2_CID_USER_NPCM_BASE + 1)
#endif
