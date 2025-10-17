/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 14, 2023.
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
#include <console/video_console.h>
#include <pexpert/GearImage.h>

struct boot_progress_element {
	unsigned int        width;
	unsigned int        height;
	int                 yOffset;
	unsigned int        res[5];
	unsigned char       data[0];
};
typedef struct boot_progress_element boot_progress_element;

static const unsigned char * default_noroot_data;

static const unsigned char * default_progress_data1x = gGearPict;
static const unsigned char * default_progress_data2x = gGearPict2x;
#if !PEXPERT_NO_3X_IMAGES
static const unsigned char * default_progress_data3x = gGearPict3x;
#else
static const unsigned char * default_progress_data3x = NULL;
#endif

static vc_progress_element default_progress =
{   0, 4 | 1, 1000 / kGearFPS, kGearFrames, {0, 0, 0},
    kGearWidth, kGearHeight, 0, kGearOffset,
    0, {0, 0, 0} };

static vc_progress_element default_noroot =
{   0, 1, 0, 0, {0, 0, 0},
    128, 128, 0, 0,
    -1, {0, 0, 0} };
