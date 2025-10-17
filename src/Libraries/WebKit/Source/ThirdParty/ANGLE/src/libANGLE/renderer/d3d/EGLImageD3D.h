/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 20, 2022.
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
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// EGLImageD3D.h: Defines the rx::EGLImageD3D class, the D3D implementation of EGL images

#ifndef LIBANGLE_RENDERER_D3D_EGLIMAGED3D_H_
#define LIBANGLE_RENDERER_D3D_EGLIMAGED3D_H_

#include "libANGLE/renderer/ImageImpl.h"

namespace gl
{
class Context;
}

namespace egl
{
class AttributeMap;
}

namespace rx
{
class FramebufferAttachmentObjectImpl;
class TextureD3D;
class RenderbufferD3D;
class RendererD3D;
class RenderTargetD3D;

class EGLImageD3D final : public ImageImpl
{
  public:
    EGLImageD3D(const egl::ImageState &state,
                EGLenum target,
                const egl::AttributeMap &attribs,
                RendererD3D *renderer);
    ~EGLImageD3D() override;

    egl::Error initialize(const egl::Display *display) override;

    angle::Result orphan(const gl::Context *context, egl::ImageSibling *sibling) override;

    angle::Result getRenderTarget(const gl::Context *context, RenderTargetD3D **outRT) const;

  private:
    angle::Result copyToLocalRendertarget(const gl::Context *context);

    RendererD3D *mRenderer;
    RenderTargetD3D *mRenderTarget;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_D3D_EGLIMAGED3D_H_
