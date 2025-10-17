/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 2, 2024.
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
#ifndef _UAPI_VGEM_DRM_H_
#define _UAPI_VGEM_DRM_H_
#include "drm.h"
#ifdef __cplusplus
extern "C" {
#endif
#define DRM_VGEM_FENCE_ATTACH 0x1
#define DRM_VGEM_FENCE_SIGNAL 0x2
#define DRM_IOCTL_VGEM_FENCE_ATTACH DRM_IOWR(DRM_COMMAND_BASE + DRM_VGEM_FENCE_ATTACH, struct drm_vgem_fence_attach)
#define DRM_IOCTL_VGEM_FENCE_SIGNAL DRM_IOW(DRM_COMMAND_BASE + DRM_VGEM_FENCE_SIGNAL, struct drm_vgem_fence_signal)
struct drm_vgem_fence_attach {
  __u32 handle;
  __u32 flags;
#define VGEM_FENCE_WRITE 0x1
  __u32 out_fence;
  __u32 pad;
};
struct drm_vgem_fence_signal {
  __u32 fence;
  __u32 flags;
};
#ifdef __cplusplus
}
#endif
#endif
