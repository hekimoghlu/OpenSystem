/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 25, 2022.
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

// WindowSurfaceEGL.h: EGL implementation of egl::Surface for windows

#ifndef LIBANGLE_RENDERER_GL_EGL_WINDOWSURFACEEGL_H_
#define LIBANGLE_RENDERER_GL_EGL_WINDOWSURFACEEGL_H_

#include "libANGLE/renderer/gl/egl/SurfaceEGL.h"

namespace rx
{

class WindowSurfaceEGL : public SurfaceEGL
{
  public:
    WindowSurfaceEGL(const egl::SurfaceState &state,
                     const FunctionsEGL *egl,
                     EGLConfig config,
                     EGLNativeWindowType window);
    ~WindowSurfaceEGL() override;

    egl::Error initialize(const egl::Display *display) override;

    egl::Error getBufferAge(const gl::Context *context, EGLint *age) override;

  private:
    EGLNativeWindowType mWindow;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_EGL_WINDOWSURFACEEGL_H_
