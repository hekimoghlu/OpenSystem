/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 25, 2022.
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

// SurfaceEGL.h: common interface for EGL surfaces

#ifndef LIBANGLE_RENDERER_GL_EGL_SURFACEEGL_H_
#define LIBANGLE_RENDERER_GL_EGL_SURFACEEGL_H_

#include <EGL/egl.h>

#include "libANGLE/renderer/gl/SurfaceGL.h"
#include "libANGLE/renderer/gl/egl/FunctionsEGL.h"

namespace rx
{

class SurfaceEGL : public SurfaceGL
{
  public:
    SurfaceEGL(const egl::SurfaceState &state, const FunctionsEGL *egl, EGLConfig config);
    ~SurfaceEGL() override;

    egl::Error makeCurrent(const gl::Context *context) override;
    egl::Error swap(const gl::Context *context) override;
    egl::Error swapWithDamage(const gl::Context *context,
                              const EGLint *rects,
                              EGLint n_rects) override;
    egl::Error postSubBuffer(const gl::Context *context,
                             EGLint x,
                             EGLint y,
                             EGLint width,
                             EGLint height) override;
    egl::Error setPresentationTime(EGLnsecsANDROID time) override;
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

    void setTimestampsEnabled(bool enabled) override;
    egl::SupportedCompositorTimings getSupportedCompositorTimings() const override;
    egl::Error getCompositorTiming(EGLint numTimestamps,
                                   const EGLint *names,
                                   EGLnsecsANDROID *values) const override;
    egl::Error getNextFrameId(EGLuint64KHR *frameId) const override;
    egl::SupportedTimestamps getSupportedTimestamps() const override;
    egl::Error getFrameTimestamps(EGLuint64KHR frameId,
                                  EGLint numTimestamps,
                                  const EGLint *timestamps,
                                  EGLnsecsANDROID *values) const override;

    EGLSurface getSurface() const;
    virtual bool isExternal() const;

  protected:
    const FunctionsEGL *mEGL;
    EGLConfig mConfig;
    EGLSurface mSurface;

  private:
    bool mHasSwapBuffersWithDamage;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_EGL_SURFACEEGL_H_
