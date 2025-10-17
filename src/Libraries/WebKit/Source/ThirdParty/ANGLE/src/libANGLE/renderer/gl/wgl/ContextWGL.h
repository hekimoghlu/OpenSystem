/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 25, 2025.
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

// ContextWGL.h: Context class for GL on Windows.  Wraps a RendererWGL.

#ifndef LIBANGLE_RENDERER_GL_WGL_CONTEXTWGL_H_
#define LIBANGLE_RENDERER_GL_WGL_CONTEXTWGL_H_

#include "libANGLE/renderer/gl/ContextGL.h"
#include "libANGLE/renderer/gl/wgl/RendererWGL.h"

namespace rx
{
class ContextWGL : public ContextGL
{
  public:
    ContextWGL(const gl::State &state,
               gl::ErrorSet *errorSet,
               const std::shared_ptr<RendererWGL> &renderer);
    ~ContextWGL() override;

    HGLRC getContext() const;

  private:
    std::shared_ptr<RendererWGL> mRendererWGL;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_WGL_RENDERERWGL_H_
