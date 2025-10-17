/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 7, 2025.
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
#ifndef __LIMA_DRM_H__
#define __LIMA_DRM_H__
#include "drm.h"
#ifdef __cplusplus
extern "C" {
#endif
enum drm_lima_param_gpu_id {
  DRM_LIMA_PARAM_GPU_ID_UNKNOWN,
  DRM_LIMA_PARAM_GPU_ID_MALI400,
  DRM_LIMA_PARAM_GPU_ID_MALI450,
};
enum drm_lima_param {
  DRM_LIMA_PARAM_GPU_ID,
  DRM_LIMA_PARAM_NUM_PP,
  DRM_LIMA_PARAM_GP_VERSION,
  DRM_LIMA_PARAM_PP_VERSION,
};
struct drm_lima_get_param {
  __u32 param;
  __u32 pad;
  __u64 value;
};
#define LIMA_BO_FLAG_HEAP (1 << 0)
struct drm_lima_gem_create {
  __u32 size;
  __u32 flags;
  __u32 handle;
  __u32 pad;
};
struct drm_lima_gem_info {
  __u32 handle;
  __u32 va;
  __u64 offset;
};
#define LIMA_SUBMIT_BO_READ 0x01
#define LIMA_SUBMIT_BO_WRITE 0x02
struct drm_lima_gem_submit_bo {
  __u32 handle;
  __u32 flags;
};
#define LIMA_GP_FRAME_REG_NUM 6
struct drm_lima_gp_frame {
  __u32 frame[LIMA_GP_FRAME_REG_NUM];
};
#define LIMA_PP_FRAME_REG_NUM 23
#define LIMA_PP_WB_REG_NUM 12
struct drm_lima_m400_pp_frame {
  __u32 frame[LIMA_PP_FRAME_REG_NUM];
  __u32 num_pp;
  __u32 wb[3 * LIMA_PP_WB_REG_NUM];
  __u32 plbu_array_address[4];
  __u32 fragment_stack_address[4];
};
struct drm_lima_m450_pp_frame {
  __u32 frame[LIMA_PP_FRAME_REG_NUM];
  __u32 num_pp;
  __u32 wb[3 * LIMA_PP_WB_REG_NUM];
  __u32 use_dlbu;
  __u32 _pad;
  union {
    __u32 plbu_array_address[8];
    __u32 dlbu_regs[4];
  };
  __u32 fragment_stack_address[8];
};
#define LIMA_PIPE_GP 0x00
#define LIMA_PIPE_PP 0x01
#define LIMA_SUBMIT_FLAG_EXPLICIT_FENCE (1 << 0)
struct drm_lima_gem_submit {
  __u32 ctx;
  __u32 pipe;
  __u32 nr_bos;
  __u32 frame_size;
  __u64 bos;
  __u64 frame;
  __u32 flags;
  __u32 out_sync;
  __u32 in_sync[2];
};
#define LIMA_GEM_WAIT_READ 0x01
#define LIMA_GEM_WAIT_WRITE 0x02
struct drm_lima_gem_wait {
  __u32 handle;
  __u32 op;
  __s64 timeout_ns;
};
struct drm_lima_ctx_create {
  __u32 id;
  __u32 _pad;
};
struct drm_lima_ctx_free {
  __u32 id;
  __u32 _pad;
};
#define DRM_LIMA_GET_PARAM 0x00
#define DRM_LIMA_GEM_CREATE 0x01
#define DRM_LIMA_GEM_INFO 0x02
#define DRM_LIMA_GEM_SUBMIT 0x03
#define DRM_LIMA_GEM_WAIT 0x04
#define DRM_LIMA_CTX_CREATE 0x05
#define DRM_LIMA_CTX_FREE 0x06
#define DRM_IOCTL_LIMA_GET_PARAM DRM_IOWR(DRM_COMMAND_BASE + DRM_LIMA_GET_PARAM, struct drm_lima_get_param)
#define DRM_IOCTL_LIMA_GEM_CREATE DRM_IOWR(DRM_COMMAND_BASE + DRM_LIMA_GEM_CREATE, struct drm_lima_gem_create)
#define DRM_IOCTL_LIMA_GEM_INFO DRM_IOWR(DRM_COMMAND_BASE + DRM_LIMA_GEM_INFO, struct drm_lima_gem_info)
#define DRM_IOCTL_LIMA_GEM_SUBMIT DRM_IOW(DRM_COMMAND_BASE + DRM_LIMA_GEM_SUBMIT, struct drm_lima_gem_submit)
#define DRM_IOCTL_LIMA_GEM_WAIT DRM_IOW(DRM_COMMAND_BASE + DRM_LIMA_GEM_WAIT, struct drm_lima_gem_wait)
#define DRM_IOCTL_LIMA_CTX_CREATE DRM_IOR(DRM_COMMAND_BASE + DRM_LIMA_CTX_CREATE, struct drm_lima_ctx_create)
#define DRM_IOCTL_LIMA_CTX_FREE DRM_IOW(DRM_COMMAND_BASE + DRM_LIMA_CTX_FREE, struct drm_lima_ctx_free)
#ifdef __cplusplus
}
#endif
#endif
