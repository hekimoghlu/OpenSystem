/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 27, 2022.
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

// SyncGL.cpp: Implements the class methods for SyncGL.

#include "libANGLE/renderer/gl/SyncGL.h"

#include "common/debug.h"
#include "libANGLE/Context.h"
#include "libANGLE/renderer/gl/ContextGL.h"
#include "libANGLE/renderer/gl/FunctionsGL.h"

namespace rx
{

SyncGL::SyncGL(const FunctionsGL *functions) : SyncImpl(), mFunctions(functions), mSyncObject(0)
{
    ASSERT(mFunctions);
}

SyncGL::~SyncGL()
{
    ASSERT(mSyncObject == 0);
}

void SyncGL::onDestroy(const gl::Context *context)
{
    ASSERT(mSyncObject != 0);
    mFunctions->deleteSync(mSyncObject);
    mSyncObject = 0;
}

angle::Result SyncGL::set(const gl::Context *context, GLenum condition, GLbitfield flags)
{
    ASSERT(condition == GL_SYNC_GPU_COMMANDS_COMPLETE && flags == 0);
    ContextGL *contextGL = GetImplAs<ContextGL>(context);
    mSyncObject          = mFunctions->fenceSync(condition, flags);
    ANGLE_CHECK(contextGL, mSyncObject != 0, "glFenceSync failed to create a GLsync object.",
                GL_OUT_OF_MEMORY);
    contextGL->markWorkSubmitted();
    return angle::Result::Continue;
}

angle::Result SyncGL::clientWait(const gl::Context *context,
                                 GLbitfield flags,
                                 GLuint64 timeout,
                                 GLenum *outResult)
{
    ASSERT(mSyncObject != 0);
    *outResult = mFunctions->clientWaitSync(mSyncObject, flags, timeout);
    return angle::Result::Continue;
}

angle::Result SyncGL::serverWait(const gl::Context *context, GLbitfield flags, GLuint64 timeout)
{
    ASSERT(mSyncObject != 0);
    mFunctions->waitSync(mSyncObject, flags, timeout);
    return angle::Result::Continue;
}

angle::Result SyncGL::getStatus(const gl::Context *context, GLint *outResult)
{
    ASSERT(mSyncObject != 0);
    mFunctions->getSynciv(mSyncObject, GL_SYNC_STATUS, 1, nullptr, outResult);
    return angle::Result::Continue;
}
}  // namespace rx
