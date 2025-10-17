/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 29, 2024.
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
#ifndef QXL_DRM_H
#define QXL_DRM_H
#include "drm.h"
#ifdef __cplusplus
extern "C" {
#endif
#define QXL_GEM_DOMAIN_CPU 0
#define QXL_GEM_DOMAIN_VRAM 1
#define QXL_GEM_DOMAIN_SURFACE 2
#define DRM_QXL_ALLOC 0x00
#define DRM_QXL_MAP 0x01
#define DRM_QXL_EXECBUFFER 0x02
#define DRM_QXL_UPDATE_AREA 0x03
#define DRM_QXL_GETPARAM 0x04
#define DRM_QXL_CLIENTCAP 0x05
#define DRM_QXL_ALLOC_SURF 0x06
struct drm_qxl_alloc {
  __u32 size;
  __u32 handle;
};
struct drm_qxl_map {
  __u64 offset;
  __u32 handle;
  __u32 pad;
};
#define QXL_RELOC_TYPE_BO 1
#define QXL_RELOC_TYPE_SURF 2
struct drm_qxl_reloc {
  __u64 src_offset;
  __u64 dst_offset;
  __u32 src_handle;
  __u32 dst_handle;
  __u32 reloc_type;
  __u32 pad;
};
struct drm_qxl_command {
  __u64 command;
  __u64 relocs;
  __u32 type;
  __u32 command_size;
  __u32 relocs_num;
  __u32 pad;
};
struct drm_qxl_execbuffer {
  __u32 flags;
  __u32 commands_num;
  __u64 commands;
};
struct drm_qxl_update_area {
  __u32 handle;
  __u32 top;
  __u32 left;
  __u32 bottom;
  __u32 right;
  __u32 pad;
};
#define QXL_PARAM_NUM_SURFACES 1
#define QXL_PARAM_MAX_RELOCS 2
struct drm_qxl_getparam {
  __u64 param;
  __u64 value;
};
struct drm_qxl_clientcap {
  __u32 index;
  __u32 pad;
};
struct drm_qxl_alloc_surf {
  __u32 format;
  __u32 width;
  __u32 height;
  __s32 stride;
  __u32 handle;
  __u32 pad;
};
#define DRM_IOCTL_QXL_ALLOC DRM_IOWR(DRM_COMMAND_BASE + DRM_QXL_ALLOC, struct drm_qxl_alloc)
#define DRM_IOCTL_QXL_MAP DRM_IOWR(DRM_COMMAND_BASE + DRM_QXL_MAP, struct drm_qxl_map)
#define DRM_IOCTL_QXL_EXECBUFFER DRM_IOW(DRM_COMMAND_BASE + DRM_QXL_EXECBUFFER, struct drm_qxl_execbuffer)
#define DRM_IOCTL_QXL_UPDATE_AREA DRM_IOW(DRM_COMMAND_BASE + DRM_QXL_UPDATE_AREA, struct drm_qxl_update_area)
#define DRM_IOCTL_QXL_GETPARAM DRM_IOWR(DRM_COMMAND_BASE + DRM_QXL_GETPARAM, struct drm_qxl_getparam)
#define DRM_IOCTL_QXL_CLIENTCAP DRM_IOW(DRM_COMMAND_BASE + DRM_QXL_CLIENTCAP, struct drm_qxl_clientcap)
#define DRM_IOCTL_QXL_ALLOC_SURF DRM_IOWR(DRM_COMMAND_BASE + DRM_QXL_ALLOC_SURF, struct drm_qxl_alloc_surf)
#ifdef __cplusplus
}
#endif
#endif
