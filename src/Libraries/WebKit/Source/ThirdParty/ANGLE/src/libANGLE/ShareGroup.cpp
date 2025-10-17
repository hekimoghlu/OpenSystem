/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 9, 2022.
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
// Copyright 2023 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// ShareGroup.h: Defines the egl::ShareGroup class, representing the collection of contexts in a
// share group.

#include "libANGLE/ShareGroup.h"

#include <algorithm>
#include <iterator>

#include <EGL/eglext.h>
#include <platform/PlatformMethods.h>

#include "common/debug.h"
#include "common/platform_helpers.h"
#include "libANGLE/Context.h"
#include "libANGLE/capture/FrameCapture.h"
#include "libANGLE/renderer/DisplayImpl.h"
#include "libANGLE/renderer/ShareGroupImpl.h"

namespace egl
{
// ShareGroupState
ShareGroupState::ShareGroupState()
    : mAnyContextWithRobustness(false), mAnyContextWithDisplayTextureShareGroup(false)
{}
ShareGroupState::~ShareGroupState() = default;

void ShareGroupState::addSharedContext(gl::Context *context)
{
    mContexts.insert(std::pair(context->id().value, context));

    if (context->isRobustnessEnabled())
    {
        mAnyContextWithRobustness = true;
    }
    if (context->getState().hasDisplayTextureShareGroup())
    {
        mAnyContextWithDisplayTextureShareGroup = true;
    }
}

void ShareGroupState::removeSharedContext(gl::Context *context)
{
    mContexts.erase(context->id().value);
}

// ShareGroup
ShareGroup::ShareGroup(rx::EGLImplFactory *factory)
    : mRefCount(1),
      mImplementation(factory->createShareGroup(mState)),
      mFrameCaptureShared(new angle::FrameCaptureShared)
{}

ShareGroup::~ShareGroup()
{
    SafeDelete(mImplementation);
}

void ShareGroup::addRef()
{
    // This is protected by global lock, so no atomic is required
    mRefCount++;
}

void ShareGroup::release(const Display *display)
{
    if (--mRefCount == 0)
    {
        if (mImplementation)
        {
            mImplementation->onDestroy(display);
        }
        delete this;
    }
}

void ShareGroup::finishAllContexts()
{
    for (auto shareContext : mState.getContexts())
    {
        if (shareContext.second->hasBeenCurrent() && !shareContext.second->isDestroyed())
        {
            shareContext.second->finish();
        }
    }
}

void ShareGroup::addSharedContext(gl::Context *context)
{
    mState.addSharedContext(context);
    mImplementation->onContextAdd();
}

void ShareGroup::removeSharedContext(gl::Context *context)
{
    mState.removeSharedContext(context);
}
}  // namespace egl
