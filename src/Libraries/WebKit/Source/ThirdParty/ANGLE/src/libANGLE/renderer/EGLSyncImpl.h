/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 13, 2024.
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
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// EGLSyncImpl.h: Defines the rx::EGLSyncImpl class.

#ifndef LIBANGLE_RENDERER_EGLSYNCIMPL_H_
#define LIBANGLE_RENDERER_EGLSYNCIMPL_H_

#include "libANGLE/Error.h"

#include "common/angleutils.h"

#include "angle_gl.h"

namespace egl
{
class AttributeMap;
class Display;
}  // namespace egl

namespace gl
{
class Context;
}  // namespace gl

namespace rx
{
class EGLSyncImpl : angle::NonCopyable
{
  public:
    EGLSyncImpl() {}
    virtual ~EGLSyncImpl() {}

    virtual void onDestroy(const egl::Display *display) {}

    virtual egl::Error initialize(const egl::Display *display,
                                  const gl::Context *context,
                                  EGLenum type,
                                  const egl::AttributeMap &attribs) = 0;
    virtual egl::Error clientWait(const egl::Display *display,
                                  const gl::Context *context,
                                  EGLint flags,
                                  EGLTime timeout,
                                  EGLint *outResult)                = 0;
    virtual egl::Error serverWait(const egl::Display *display,
                                  const gl::Context *context,
                                  EGLint flags)                     = 0;
    virtual egl::Error signal(const egl::Display *display, const gl::Context *context, EGLint mode);
    virtual egl::Error getStatus(const egl::Display *display, EGLint *outStatus) = 0;
    virtual egl::Error copyMetalSharedEventANGLE(const egl::Display *display,
                                                 void **outEvent) const;
    virtual egl::Error dupNativeFenceFD(const egl::Display *display, EGLint *fdOut) const;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_EGLSYNCIMPL_H_
