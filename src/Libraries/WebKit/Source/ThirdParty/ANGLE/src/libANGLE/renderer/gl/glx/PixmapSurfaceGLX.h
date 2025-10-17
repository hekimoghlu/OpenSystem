/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 16, 2025.
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
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// PixmapSurfaceGLX.h: GLX implementation of egl::Surface for Pixmaps

#ifndef LIBANGLE_RENDERER_GL_GLX_PIXMAPSURFACEGLX_H_
#define LIBANGLE_RENDERER_GL_GLX_PIXMAPSURFACEGLX_H_

#include "libANGLE/renderer/gl/glx/SurfaceGLX.h"
#include "libANGLE/renderer/gl/glx/platform_glx.h"

namespace rx
{

class FunctionsGLX;

class PixmapSurfaceGLX : public SurfaceGLX
{
  public:
    PixmapSurfaceGLX(const egl::SurfaceState &state,
                     Pixmap pixmap,
                     Display *display,
                     const FunctionsGLX &glx,
                     glx::FBConfig fbConfig);
    ~PixmapSurfaceGLX() override;

    egl::Error initialize(const egl::Display *display) override;
    egl::Error makeCurrent(const gl::Context *context) override;

    egl::Error swap(const gl::Context *context) override;
    egl::Error postSubBuffer(const gl::Context *context,
                             EGLint x,
                             EGLint y,
                             EGLint width,
                             EGLint height) override;
    egl::Error querySurfacePointerANGLE(EGLint attribute, void **value) override;
    egl::Error bindTexImage(const gl::Context *context,
                            gl::Texture *texture,
                            EGLint buffer) override;
    egl::Error releaseTexImage(const gl::Context *context, EGLint buffer) override;
    void setSwapInterval(const egl::Display *display, EGLint interval) override;

    EGLint getWidth() const override;
    EGLint getHeight() const override;

    EGLint isPostSubBufferSupported() const override;
    EGLint getSwapBehavior() const override;

    egl::Error checkForResize() override;
    glx::Drawable getDrawable() const override;

  private:
    unsigned mWidth;
    unsigned mHeight;

    const FunctionsGLX &mGLX;
    glx::FBConfig mFBConfig;
    Pixmap mXPixmap;
    glx::Pixmap mGLXPixmap;
    Display *mDisplay;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_GLX_PIXMAPSURFACEGLX_H_
