/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 25, 2022.
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
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// ContextEGL.h: Context class for GL on Android/ChromeOS.  Wraps a RendererEGL.

#ifndef LIBANGLE_RENDERER_GL_EGL_CONTEXTEGL_H_
#define LIBANGLE_RENDERER_GL_EGL_CONTEXTEGL_H_

#include "libANGLE/renderer/gl/ContextGL.h"
#include "libANGLE/renderer/gl/egl/RendererEGL.h"

namespace rx
{

struct ExternalContextState;

class ContextEGL : public ContextGL
{
  public:
    ContextEGL(const gl::State &state,
               gl::ErrorSet *errorSet,
               const std::shared_ptr<RendererEGL> &renderer,
               RobustnessVideoMemoryPurgeStatus robustnessVideoMemoryPurgeStatus);
    ~ContextEGL() override;

    void acquireExternalContext(const gl::Context *context) override;
    void releaseExternalContext(const gl::Context *context) override;

    EGLContext getContext() const;

  private:
    std::shared_ptr<RendererEGL> mRendererEGL;
    std::unique_ptr<ExternalContextState> mExtState;

    // Used to restore the default FBO's ID on unmaking an external context
    // current, as when making an external context current ANGLE sets the
    // default FBO's ID to that bound in the external context.
    GLuint mPrevDefaultFramebufferID = 0;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_EGL_RENDEREREGL_H_
