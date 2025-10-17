/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 17, 2022.
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
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// EGLReusableSync.cpp: Implements the egl::ReusableSync class.

#include "libANGLE/renderer/EGLReusableSync.h"

#include "libANGLE/Context.h"
#include "libANGLE/renderer/ContextImpl.h"

namespace rx
{

ReusableSync::ReusableSync() : EGLSyncImpl(), mStatus(0) {}

void ReusableSync::onDestroy(const egl::Display *display) {}

ReusableSync::~ReusableSync()
{
    // Release any waiting thread.
    mCondVar.notify_all();
}

egl::Error ReusableSync::initialize(const egl::Display *display,
                                    const gl::Context *context,
                                    EGLenum type,
                                    const egl::AttributeMap &attribs)
{
    ASSERT(type == EGL_SYNC_REUSABLE_KHR);
    mStatus = EGL_UNSIGNALED;
    return egl::NoError();
}

egl::Error ReusableSync::clientWait(const egl::Display *display,
                                    const gl::Context *context,
                                    EGLint flags,
                                    EGLTime timeout,
                                    EGLint *outResult)
{
    if (mStatus == EGL_SIGNALED)
    {
        *outResult = EGL_CONDITION_SATISFIED_KHR;
        return egl::NoError();
    }
    if (((flags & EGL_SYNC_FLUSH_COMMANDS_BIT) != 0) && (context != nullptr))
    {
        angle::Result result = context->getImplementation()->flush(context);
        if (result != angle::Result::Continue)
        {
            return ResultToEGL(result);
        }
    }
    if (timeout == 0)
    {
        *outResult = EGL_TIMEOUT_EXPIRED_KHR;
        return egl::NoError();
    }

    using NanoSeconds    = std::chrono::duration<int64_t, std::nano>;
    NanoSeconds duration = (timeout == EGL_FOREVER) ? NanoSeconds::max() : NanoSeconds(timeout);
    std::cv_status waitStatus = std::cv_status::no_timeout;
    mMutex.lock();
    waitStatus = mCondVar.wait_for(mMutex, duration);
    mMutex.unlock();

    switch (waitStatus)
    {
        case std::cv_status::no_timeout:  // Signaled.
            *outResult = EGL_CONDITION_SATISFIED_KHR;
            break;
        case std::cv_status::timeout:  // Timed-out.
            *outResult = EGL_TIMEOUT_EXPIRED_KHR;
            break;
        default:
            break;
    }
    return egl::NoError();
}

egl::Error ReusableSync::serverWait(const egl::Display *display,
                                    const gl::Context *context,
                                    EGLint flags)
{
    // Does not support server wait.
    return egl::EglBadMatch();
}

egl::Error ReusableSync::signal(const egl::Display *display,
                                const gl::Context *context,
                                EGLint mode)
{
    if (mode == EGL_SIGNALED)
    {
        if (mStatus == EGL_UNSIGNALED)
        {
            // Release all threads.
            mCondVar.notify_all();
        }
        mStatus = EGL_SIGNALED;
    }
    else
    {
        mStatus = EGL_UNSIGNALED;
    }
    return egl::NoError();
}

egl::Error ReusableSync::getStatus(const egl::Display *display, EGLint *outStatus)
{
    *outStatus = mStatus;
    return egl::NoError();
}

}  // namespace rx
