/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 29, 2023.
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
#ifndef __OMAP_DRM_H__
#define __OMAP_DRM_H__
#include "drm.h"
#ifdef __cplusplus
extern "C" {
#endif
#define OMAP_PARAM_CHIPSET_ID 1
struct drm_omap_param {
  __u64 param;
  __u64 value;
};
#define OMAP_BO_SCANOUT 0x00000001
#define OMAP_BO_CACHED 0x00000000
#define OMAP_BO_WC 0x00000002
#define OMAP_BO_UNCACHED 0x00000004
#define OMAP_BO_CACHE_MASK 0x00000006
#define OMAP_BO_TILED_8 0x00000100
#define OMAP_BO_TILED_16 0x00000200
#define OMAP_BO_TILED_32 0x00000300
#define OMAP_BO_TILED_MASK 0x00000f00
union omap_gem_size {
  __u32 bytes;
  struct {
    __u16 width;
    __u16 height;
  } tiled;
};
struct drm_omap_gem_new {
  union omap_gem_size size;
  __u32 flags;
  __u32 handle;
  __u32 __pad;
};
enum omap_gem_op {
  OMAP_GEM_READ = 0x01,
  OMAP_GEM_WRITE = 0x02,
};
struct drm_omap_gem_cpu_prep {
  __u32 handle;
  __u32 op;
};
struct drm_omap_gem_cpu_fini {
  __u32 handle;
  __u32 op;
  __u32 nregions;
  __u32 __pad;
};
struct drm_omap_gem_info {
  __u32 handle;
  __u32 pad;
  __u64 offset;
  __u32 size;
  __u32 __pad;
};
#define DRM_OMAP_GET_PARAM 0x00
#define DRM_OMAP_SET_PARAM 0x01
#define DRM_OMAP_GEM_NEW 0x03
#define DRM_OMAP_GEM_CPU_PREP 0x04
#define DRM_OMAP_GEM_CPU_FINI 0x05
#define DRM_OMAP_GEM_INFO 0x06
#define DRM_OMAP_NUM_IOCTLS 0x07
#define DRM_IOCTL_OMAP_GET_PARAM DRM_IOWR(DRM_COMMAND_BASE + DRM_OMAP_GET_PARAM, struct drm_omap_param)
#define DRM_IOCTL_OMAP_SET_PARAM DRM_IOW(DRM_COMMAND_BASE + DRM_OMAP_SET_PARAM, struct drm_omap_param)
#define DRM_IOCTL_OMAP_GEM_NEW DRM_IOWR(DRM_COMMAND_BASE + DRM_OMAP_GEM_NEW, struct drm_omap_gem_new)
#define DRM_IOCTL_OMAP_GEM_CPU_PREP DRM_IOW(DRM_COMMAND_BASE + DRM_OMAP_GEM_CPU_PREP, struct drm_omap_gem_cpu_prep)
#define DRM_IOCTL_OMAP_GEM_CPU_FINI DRM_IOW(DRM_COMMAND_BASE + DRM_OMAP_GEM_CPU_FINI, struct drm_omap_gem_cpu_fini)
#define DRM_IOCTL_OMAP_GEM_INFO DRM_IOWR(DRM_COMMAND_BASE + DRM_OMAP_GEM_INFO, struct drm_omap_gem_info)
#ifdef __cplusplus
}
#endif
#endif
