/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 15, 2024.
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

// EGLImageD3D.cpp: Implements the rx::EGLImageD3D class, the D3D implementation of EGL images

#include "libANGLE/renderer/d3d/EGLImageD3D.h"

#include "common/debug.h"
#include "common/utilities.h"
#include "libANGLE/AttributeMap.h"
#include "libANGLE/Context.h"
#include "libANGLE/Texture.h"
#include "libANGLE/renderer/d3d/RenderTargetD3D.h"
#include "libANGLE/renderer/d3d/RenderbufferD3D.h"
#include "libANGLE/renderer/d3d/RendererD3D.h"
#include "libANGLE/renderer/d3d/TextureD3D.h"
#include "libANGLE/renderer/d3d/TextureStorage.h"

#include <EGL/eglext.h>

namespace rx
{

EGLImageD3D::EGLImageD3D(const egl::ImageState &state,
                         EGLenum target,
                         const egl::AttributeMap &attribs,
                         RendererD3D *renderer)
    : ImageImpl(state), mRenderer(renderer), mRenderTarget(nullptr)
{
    ASSERT(renderer != nullptr);
}

EGLImageD3D::~EGLImageD3D()
{
    SafeDelete(mRenderTarget);
}

egl::Error EGLImageD3D::initialize(const egl::Display *display)
{
    return egl::NoError();
}

angle::Result EGLImageD3D::orphan(const gl::Context *context, egl::ImageSibling *sibling)
{
    if (sibling == mState.source)
    {
        ANGLE_TRY(copyToLocalRendertarget(context));
    }

    return angle::Result::Continue;
}

angle::Result EGLImageD3D::getRenderTarget(const gl::Context *context,
                                           RenderTargetD3D **outRT) const
{
    if (mState.source != nullptr)
    {
        ASSERT(!mRenderTarget);
        FramebufferAttachmentRenderTarget *rt = nullptr;
        ANGLE_TRY(
            mState.source->getAttachmentRenderTarget(context, GL_NONE, mState.imageIndex, 0, &rt));
        *outRT = static_cast<RenderTargetD3D *>(rt);
        return angle::Result::Continue;
    }

    ASSERT(mRenderTarget);
    *outRT = mRenderTarget;
    return angle::Result::Continue;
}

angle::Result EGLImageD3D::copyToLocalRendertarget(const gl::Context *context)
{
    ASSERT(mState.source != nullptr);
    ASSERT(mRenderTarget == nullptr);

    RenderTargetD3D *curRenderTarget = nullptr;
    ANGLE_TRY(getRenderTarget(context, &curRenderTarget));

    {
        std::unique_lock lock(mState.targetsLock);
        // Invalidate FBOs with this Image attached. Only currently applies to D3D11.
        for (egl::ImageSibling *target : mState.targets)
        {
            target->onStateChange(angle::SubjectMessage::SubjectChanged);
        }
    }

    return mRenderer->createRenderTargetCopy(context, curRenderTarget, &mRenderTarget);
}
}  // namespace rx
