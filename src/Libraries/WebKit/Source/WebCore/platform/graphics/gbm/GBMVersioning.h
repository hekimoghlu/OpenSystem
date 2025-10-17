/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 10, 2022.
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
#pragma once

#if USE(GBM)

#include <gbm.h>

#if !HAVE(GBM_BO_GET_FD_FOR_PLANE)
#include <fcntl.h>
#include <xf86drm.h>
#endif

#if !HAVE(GBM_BO_CREATE_WITH_MODIFIERS2)
static inline struct gbm_bo* gbm_bo_create_with_modifiers2(struct gbm_device* gbm, uint32_t width, uint32_t height, uint32_t format, const uint64_t* modifiers, const unsigned count, uint32_t)
{
    return gbm_bo_create_with_modifiers(gbm, width, height, format, modifiers, count);
}
#endif

#if !HAVE(GBM_BO_GET_FD_FOR_PLANE)
static inline int gbm_bo_get_fd_for_plane(struct gbm_bo* bo, int plane)
{
    auto handle = gbm_bo_get_handle_for_plane(bo, plane);
    if (handle.s32 == -1)
        return -1;

    int fd;
    int ret = drmPrimeHandleToFD(gbm_device_get_fd(gbm_bo_get_device(bo)), handle.u32, DRM_CLOEXEC, &fd);
    return ret < 0 ? -1 : fd;
}
#endif

#endif // USE(GBM)
