/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 4, 2023.
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

// EGLSync.cpp: Implements the egl::Sync class.

#include "libANGLE/EGLSync.h"

#include "angle_gl.h"

#include "common/utilities.h"
#include "libANGLE/renderer/EGLImplFactory.h"
#include "libANGLE/renderer/EGLReusableSync.h"
#include "libANGLE/renderer/EGLSyncImpl.h"

namespace egl
{

Sync::Sync(rx::EGLImplFactory *factory, EGLenum type)
    : mLabel(nullptr), mId({0}), mType(type), mCondition(0), mNativeFenceFD(0)
{
    switch (mType)
    {
        case EGL_SYNC_FENCE:
        case EGL_SYNC_GLOBAL_FENCE_ANGLE:
        case EGL_SYNC_NATIVE_FENCE_ANDROID:
        case EGL_SYNC_METAL_SHARED_EVENT_ANGLE:
            mFence = std::unique_ptr<rx::EGLSyncImpl>(factory->createSync());
            break;

        case EGL_SYNC_REUSABLE_KHR:
            mFence = std::unique_ptr<rx::EGLSyncImpl>(new rx::ReusableSync());
            break;

        default:
            UNREACHABLE();
    }
}

void Sync::onDestroy(const Display *display)
{
    ASSERT(mFence);
    mFence->onDestroy(display);
}

Sync::~Sync() {}

Error Sync::initialize(const Display *display,
                       const gl::Context *context,
                       const SyncID &id,
                       const AttributeMap &attribs)
{
    mId           = id;
    mAttributeMap = attribs;
    mNativeFenceFD =
        attribs.getAsInt(EGL_SYNC_NATIVE_FENCE_FD_ANDROID, EGL_NO_NATIVE_FENCE_FD_ANDROID);
    mCondition = EGL_SYNC_PRIOR_COMMANDS_COMPLETE_KHR;

    // Per extension spec: Signaling Condition.
    // "If the EGL_SYNC_NATIVE_FENCE_FD_ANDROID attribute is not
    // EGL_NO_NATIVE_FENCE_FD_ANDROID then the EGL_SYNC_CONDITION_KHR attribute
    // is set to EGL_SYNC_NATIVE_FENCE_SIGNALED_ANDROID and the EGL_SYNC_STATUS_KHR
    // attribute is set to reflect the signal status of the native fence object.
    if ((mType == EGL_SYNC_NATIVE_FENCE_ANDROID) &&
        (mNativeFenceFD != EGL_NO_NATIVE_FENCE_FD_ANDROID))
    {
        mCondition = EGL_SYNC_NATIVE_FENCE_SIGNALED_ANDROID;
    }

    // Per extension spec: Signaling Condition.
    if (mType == EGL_SYNC_METAL_SHARED_EVENT_ANGLE)
    {
        mCondition = attribs.getAsInt(EGL_SYNC_CONDITION, EGL_SYNC_PRIOR_COMMANDS_COMPLETE_KHR);
    }

    return mFence->initialize(display, context, mType, mAttributeMap);
}

void Sync::setLabel(EGLLabelKHR label)
{
    mLabel = label;
}

EGLLabelKHR Sync::getLabel() const
{
    return mLabel;
}

Error Sync::clientWait(const Display *display,
                       const gl::Context *context,
                       EGLint flags,
                       EGLTime timeout,
                       EGLint *outResult)
{
    return mFence->clientWait(display, context, flags, timeout, outResult);
}

Error Sync::serverWait(const Display *display, const gl::Context *context, EGLint flags)
{
    return mFence->serverWait(display, context, flags);
}

Error Sync::signal(const Display *display, const gl::Context *context, EGLint mode)
{
    return mFence->signal(display, context, mode);
}

Error Sync::getStatus(const Display *display, EGLint *outStatus) const
{
    return mFence->getStatus(display, outStatus);
}

Error Sync::copyMetalSharedEventANGLE(const Display *display, void **result) const
{
    return mFence->copyMetalSharedEventANGLE(display, result);
}

Error Sync::dupNativeFenceFD(const Display *display, EGLint *result) const
{
    return mFence->dupNativeFenceFD(display, result);
}

}  // namespace egl
