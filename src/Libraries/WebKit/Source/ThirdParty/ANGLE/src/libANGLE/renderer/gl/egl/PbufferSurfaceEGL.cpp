/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 16, 2024.
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

// PbufferSurfaceEGL.h: EGL implementation of egl::Surface for pbuffers

#include "libANGLE/renderer/gl/egl/PbufferSurfaceEGL.h"

#include "libANGLE/Surface.h"
#include "libANGLE/renderer/gl/egl/egl_utils.h"

namespace rx
{

PbufferSurfaceEGL::PbufferSurfaceEGL(const egl::SurfaceState &state,
                                     const FunctionsEGL *egl,
                                     EGLConfig config)
    : SurfaceEGL(state, egl, config)
{}

PbufferSurfaceEGL::~PbufferSurfaceEGL() {}

egl::Error PbufferSurfaceEGL::initialize(const egl::Display *display)
{
    constexpr EGLint kForwardedPBufferSurfaceAttributes[] = {
        EGL_WIDTH,          EGL_HEIGHT,         EGL_LARGEST_PBUFFER, EGL_TEXTURE_FORMAT,
        EGL_TEXTURE_TARGET, EGL_MIPMAP_TEXTURE, EGL_VG_COLORSPACE,   EGL_VG_ALPHA_FORMAT,
    };

    native_egl::AttributeVector nativeAttribs =
        native_egl::TrimAttributeMap(mState.attributes, kForwardedPBufferSurfaceAttributes);
    native_egl::FinalizeAttributeVector(&nativeAttribs);

    mSurface = mEGL->createPbufferSurface(mConfig, nativeAttribs.data());
    if (mSurface == EGL_NO_SURFACE)
    {
        return egl::Error(mEGL->getError(), "eglCreatePbufferSurface failed");
    }

    return egl::NoError();
}

}  // namespace rx
