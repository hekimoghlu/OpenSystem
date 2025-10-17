/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 9, 2022.
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

//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// EGLImplFactory.h:
//   Factory interface for EGL Impl objects.
//

#ifndef LIBANGLE_RENDERER_EGLIMPLFACTORY_H_
#define LIBANGLE_RENDERER_EGLIMPLFACTORY_H_

#include "libANGLE/Stream.h"

namespace egl
{
class AttributeMap;
struct Config;
class ImageSibling;
struct ImageState;
class ShareGroupState;
struct SurfaceState;
}  // namespace egl

namespace gl
{
class Context;
class ErrorSet;
class State;
}  // namespace gl

namespace rx
{
class ContextImpl;
class EGLSyncImpl;
class ImageImpl;
class ExternalImageSiblingImpl;
class SurfaceImpl;
class ShareGroupImpl;

class EGLImplFactory : angle::NonCopyable
{
  public:
    EGLImplFactory() {}
    virtual ~EGLImplFactory() {}

    virtual SurfaceImpl *createWindowSurface(const egl::SurfaceState &state,
                                             EGLNativeWindowType window,
                                             const egl::AttributeMap &attribs)           = 0;
    virtual SurfaceImpl *createPbufferSurface(const egl::SurfaceState &state,
                                              const egl::AttributeMap &attribs)          = 0;
    virtual SurfaceImpl *createPbufferFromClientBuffer(const egl::SurfaceState &state,
                                                       EGLenum buftype,
                                                       EGLClientBuffer clientBuffer,
                                                       const egl::AttributeMap &attribs) = 0;
    virtual SurfaceImpl *createPixmapSurface(const egl::SurfaceState &state,
                                             NativePixmapType nativePixmap,
                                             const egl::AttributeMap &attribs)           = 0;

    virtual ImageImpl *createImage(const egl::ImageState &state,
                                   const gl::Context *context,
                                   EGLenum target,
                                   const egl::AttributeMap &attribs) = 0;

    virtual ContextImpl *createContext(const gl::State &state,
                                       gl::ErrorSet *errorSet,
                                       const egl::Config *configuration,
                                       const gl::Context *shareContext,
                                       const egl::AttributeMap &attribs) = 0;

    virtual StreamProducerImpl *createStreamProducerD3DTexture(
        egl::Stream::ConsumerType consumerType,
        const egl::AttributeMap &attribs) = 0;

    virtual ExternalImageSiblingImpl *createExternalImageSibling(const gl::Context *context,
                                                                 EGLenum target,
                                                                 EGLClientBuffer buffer,
                                                                 const egl::AttributeMap &attribs);

    virtual EGLSyncImpl *createSync();

    virtual ShareGroupImpl *createShareGroup(const egl::ShareGroupState &state) = 0;
};

inline ExternalImageSiblingImpl *EGLImplFactory::createExternalImageSibling(
    const gl::Context *context,
    EGLenum target,
    EGLClientBuffer buffer,
    const egl::AttributeMap &attribs)
{
    UNREACHABLE();
    return nullptr;
}

inline EGLSyncImpl *EGLImplFactory::createSync()
{
    UNREACHABLE();
    return nullptr;
}

}  // namespace rx

#endif  // LIBANGLE_RENDERER_EGLIMPLFACTORY_H_
