/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 12, 2023.
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

#include <xf86drm.h>
#include <xf86drmMode.h>

namespace WPE {

namespace DRM {

template<typename T>
struct PtrDeleter {
    void operator()(T* ptr) const
    {
        drmFree(ptr);
    }
};

template<typename T, typename U = PtrDeleter<T>>
using UniquePtr = std::unique_ptr<T, U>;

#define FOR_EACH_DRM_DELETER(macro) \
    macro(drmModeAtomicReq, drmModeAtomicFree) \
    macro(drmModeConnector, drmModeFreeConnector) \
    macro(drmModeCrtc, drmModeFreeCrtc) \
    macro(drmModeEncoder, drmModeFreeEncoder) \
    macro(drmModeObjectProperties, drmModeFreeObjectProperties) \
    macro(drmModePropertyBlobRes, drmModeFreePropertyBlob) \
    macro(drmModePropertyRes, drmModeFreeProperty) \
    macro(drmModePlane, drmModeFreePlane) \
    macro(drmModePlaneRes, drmModeFreePlaneResources) \
    macro(drmModeRes, drmModeFreeResources)

#define DEFINE_DRM_DELETER(typeName, deleterFunc) \
    template<> struct PtrDeleter<typeName> \
    { \
        void operator() (typeName* ptr) const \
        { \
            deleterFunc(ptr); \
        } \
    };

FOR_EACH_DRM_DELETER(DEFINE_DRM_DELETER)
#undef FOR_EACH_DRM_DELETER

} // namespace DRM

} // namespace WPE
