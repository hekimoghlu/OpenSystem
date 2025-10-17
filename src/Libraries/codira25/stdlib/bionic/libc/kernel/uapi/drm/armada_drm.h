/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 31, 2023.
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
#ifndef DRM_ARMADA_IOCTL_H
#define DRM_ARMADA_IOCTL_H
#include "drm.h"
#ifdef __cplusplus
extern "C" {
#endif
#define DRM_ARMADA_GEM_CREATE 0x00
#define DRM_ARMADA_GEM_MMAP 0x02
#define DRM_ARMADA_GEM_PWRITE 0x03
#define ARMADA_IOCTL(dir,name,str) DRM_ ##dir(DRM_COMMAND_BASE + DRM_ARMADA_ ##name, struct drm_armada_ ##str)
struct drm_armada_gem_create {
  __u32 handle;
  __u32 size;
};
#define DRM_IOCTL_ARMADA_GEM_CREATE ARMADA_IOCTL(IOWR, GEM_CREATE, gem_create)
struct drm_armada_gem_mmap {
  __u32 handle;
  __u32 pad;
  __u64 offset;
  __u64 size;
  __u64 addr;
};
#define DRM_IOCTL_ARMADA_GEM_MMAP ARMADA_IOCTL(IOWR, GEM_MMAP, gem_mmap)
struct drm_armada_gem_pwrite {
  __u64 ptr;
  __u32 handle;
  __u32 offset;
  __u32 size;
};
#define DRM_IOCTL_ARMADA_GEM_PWRITE ARMADA_IOCTL(IOW, GEM_PWRITE, gem_pwrite)
#ifdef __cplusplus
}
#endif
#endif
