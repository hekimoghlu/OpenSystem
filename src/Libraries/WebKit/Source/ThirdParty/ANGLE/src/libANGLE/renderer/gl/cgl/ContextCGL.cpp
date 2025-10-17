/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 2, 2022.
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
// ContextCGL:
//   Mac-specific subclass of ContextGL.
//

#include "libANGLE/renderer/gl/cgl/ContextCGL.h"

#include "libANGLE/Context.h"
#include "libANGLE/Display.h"
#include "libANGLE/renderer/gl/cgl/DisplayCGL.h"

namespace rx
{

ContextCGL::ContextCGL(DisplayCGL *display,
                       const gl::State &state,
                       gl::ErrorSet *errorSet,
                       const std::shared_ptr<RendererGL> &renderer,
                       bool usesDiscreteGPU)
    : ContextGL(state, errorSet, renderer, RobustnessVideoMemoryPurgeStatus::NOT_REQUESTED),
      mUsesDiscreteGpu(usesDiscreteGPU),
      mReleasedDiscreteGpu(false)
{
    if (mUsesDiscreteGpu)
    {
        (void)display->referenceDiscreteGPU();
    }
}

egl::Error ContextCGL::releaseHighPowerGPU(gl::Context *context)
{
    if (mUsesDiscreteGpu && !mReleasedDiscreteGpu)
    {
        mReleasedDiscreteGpu = true;
        return GetImplAs<DisplayCGL>(context->getDisplay())->unreferenceDiscreteGPU();
    }

    return egl::NoError();
}

egl::Error ContextCGL::reacquireHighPowerGPU(gl::Context *context)
{
    if (mUsesDiscreteGpu && mReleasedDiscreteGpu)
    {
        mReleasedDiscreteGpu = false;
        return GetImplAs<DisplayCGL>(context->getDisplay())->referenceDiscreteGPU();
    }

    return egl::NoError();
}

void ContextCGL::onDestroy(const gl::Context *context)
{
    if (mUsesDiscreteGpu && !mReleasedDiscreteGpu)
    {
        (void)GetImplAs<DisplayCGL>(context->getDisplay())->unreferenceDiscreteGPU();
    }
    ContextGL::onDestroy(context);
}

}  // namespace rx
