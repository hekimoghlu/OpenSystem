/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 25, 2023.
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

// SyncEGL.h: Defines the rx::SyncEGL class, the EGL implementation of EGL sync objects.

#ifndef LIBANGLE_RENDERER_GL_EGL_SYNCEGL_H_
#define LIBANGLE_RENDERER_GL_EGL_SYNCEGL_H_

#include "libANGLE/renderer/EGLSyncImpl.h"

namespace egl
{
class AttributeMap;
}

namespace rx
{

class FunctionsEGL;

class SyncEGL final : public EGLSyncImpl
{
  public:
    SyncEGL(const FunctionsEGL *egl);
    ~SyncEGL() override;

    void onDestroy(const egl::Display *display) override;

    egl::Error initialize(const egl::Display *display,
                          const gl::Context *context,
                          EGLenum type,
                          const egl::AttributeMap &attribs) override;
    egl::Error clientWait(const egl::Display *display,
                          const gl::Context *context,
                          EGLint flags,
                          EGLTime timeout,
                          EGLint *outResult) override;
    egl::Error serverWait(const egl::Display *display,
                          const gl::Context *context,
                          EGLint flags) override;
    egl::Error getStatus(const egl::Display *display, EGLint *outStatus) override;

    egl::Error dupNativeFenceFD(const egl::Display *display, EGLint *result) const override;

  private:
    const FunctionsEGL *mEGL;

    EGLSync mSync;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_EGL_IMAGEEGL_H_
