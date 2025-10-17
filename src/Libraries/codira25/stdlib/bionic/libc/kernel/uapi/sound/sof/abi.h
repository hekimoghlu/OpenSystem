/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 7, 2025.
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
#ifndef __INCLUDE_UAPI_SOUND_SOF_ABI_H__
#define __INCLUDE_UAPI_SOUND_SOF_ABI_H__
#include <linux/types.h>
#define SOF_ABI_MAJOR 3
#define SOF_ABI_MINOR 23
#define SOF_ABI_PATCH 1
#define SOF_ABI_MAJOR_SHIFT 24
#define SOF_ABI_MAJOR_MASK 0xff
#define SOF_ABI_MINOR_SHIFT 12
#define SOF_ABI_MINOR_MASK 0xfff
#define SOF_ABI_PATCH_SHIFT 0
#define SOF_ABI_PATCH_MASK 0xfff
#define SOF_ABI_VER(major,minor,patch) (((major) << SOF_ABI_MAJOR_SHIFT) | ((minor) << SOF_ABI_MINOR_SHIFT) | ((patch) << SOF_ABI_PATCH_SHIFT))
#define SOF_ABI_VERSION_MAJOR(version) (((version) >> SOF_ABI_MAJOR_SHIFT) & SOF_ABI_MAJOR_MASK)
#define SOF_ABI_VERSION_MINOR(version) (((version) >> SOF_ABI_MINOR_SHIFT) & SOF_ABI_MINOR_MASK)
#define SOF_ABI_VERSION_PATCH(version) (((version) >> SOF_ABI_PATCH_SHIFT) & SOF_ABI_PATCH_MASK)
#define SOF_ABI_VERSION_INCOMPATIBLE(sof_ver,client_ver) (SOF_ABI_VERSION_MAJOR((sof_ver)) != SOF_ABI_VERSION_MAJOR((client_ver)))
#define SOF_ABI_VERSION SOF_ABI_VER(SOF_ABI_MAJOR, SOF_ABI_MINOR, SOF_ABI_PATCH)
#define SOF_ABI_MAGIC 0x00464F53
#define SOF_IPC4_ABI_MAGIC 0x34464F53
#endif
