/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 12, 2024.
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
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Fence.cpp: Implements the gl::FenceNV and gl::Sync classes.

#include "libANGLE/Fence.h"

#include "angle_gl.h"

#include "common/utilities.h"
#include "libANGLE/renderer/FenceNVImpl.h"
#include "libANGLE/renderer/GLImplFactory.h"
#include "libANGLE/renderer/SyncImpl.h"

namespace gl
{

FenceNV::FenceNV(rx::GLImplFactory *factory)
    : mFence(factory->createFenceNV()), mIsSet(false), mStatus(GL_FALSE), mCondition(GL_NONE)
{}

FenceNV::~FenceNV()
{
    SafeDelete(mFence);
}

void FenceNV::onDestroy(const gl::Context *context)
{
    mFence->onDestroy(context);
}

angle::Result FenceNV::set(const Context *context, GLenum condition)
{
    ANGLE_TRY(mFence->set(context, condition));

    mCondition = condition;
    mStatus    = GL_FALSE;
    mIsSet     = true;

    return angle::Result::Continue;
}

angle::Result FenceNV::test(const Context *context, GLboolean *outResult)
{
    // Flush the command buffer by default
    ANGLE_TRY(mFence->test(context, &mStatus));

    *outResult = mStatus;
    return angle::Result::Continue;
}

angle::Result FenceNV::finish(const Context *context)
{
    ASSERT(mIsSet);

    ANGLE_TRY(mFence->finish(context));

    mStatus = GL_TRUE;

    return angle::Result::Continue;
}

Sync::Sync(rx::GLImplFactory *factory, SyncID id)
    : RefCountObject(factory->generateSerial(), id),
      mFence(factory->createSync()),
      mLabel(),
      mCondition(GL_SYNC_GPU_COMMANDS_COMPLETE),
      mFlags(0)
{}

void Sync::onDestroy(const Context *context)
{
    ASSERT(mFence);
    mFence->onDestroy(context);
}

Sync::~Sync()
{
    SafeDelete(mFence);
}

angle::Result Sync::setLabel(const Context *context, const std::string &label)
{
    mLabel = label;
    return angle::Result::Continue;
}

const std::string &Sync::getLabel() const
{
    return mLabel;
}

angle::Result Sync::set(const Context *context, GLenum condition, GLbitfield flags)
{
    ANGLE_TRY(mFence->set(context, condition, flags));

    mCondition = condition;
    mFlags     = flags;
    return angle::Result::Continue;
}

angle::Result Sync::clientWait(const Context *context,
                               GLbitfield flags,
                               GLuint64 timeout,
                               GLenum *outResult)
{
    ASSERT(mCondition != GL_NONE);
    return mFence->clientWait(context, flags, timeout, outResult);
}

angle::Result Sync::serverWait(const Context *context, GLbitfield flags, GLuint64 timeout)
{
    return mFence->serverWait(context, flags, timeout);
}

angle::Result Sync::getStatus(const Context *context, GLint *outResult) const
{
    return mFence->getStatus(context, outResult);
}

}  // namespace gl
