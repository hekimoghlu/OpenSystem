/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 26, 2022.
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
#ifndef _DRM_SAREA_H_
#define _DRM_SAREA_H_
#include "drm.h"
#ifdef __cplusplus
extern "C" {
#endif
#ifdef __alpha__
#define SAREA_MAX 0x2000U
#elif defined(__mips__)
#define SAREA_MAX 0x4000U
#elif defined(__ia64__)
#define SAREA_MAX 0x10000U
#else
#define SAREA_MAX 0x2000U
#endif
#define SAREA_MAX_DRAWABLES 256
#define SAREA_DRAWABLE_CLAIMED_ENTRY 0x80000000
struct drm_sarea_drawable {
  unsigned int stamp;
  unsigned int flags;
};
struct drm_sarea_frame {
  unsigned int x;
  unsigned int y;
  unsigned int width;
  unsigned int height;
  unsigned int fullscreen;
};
struct drm_sarea {
  struct drm_hw_lock lock;
  struct drm_hw_lock drawable_lock;
  struct drm_sarea_drawable drawableTable[SAREA_MAX_DRAWABLES];
  struct drm_sarea_frame frame;
  drm_context_t dummy_context;
};
typedef struct drm_sarea_drawable drm_sarea_drawable_t;
typedef struct drm_sarea_frame drm_sarea_frame_t;
typedef struct drm_sarea drm_sarea_t;
#ifdef __cplusplus
}
#endif
#endif
