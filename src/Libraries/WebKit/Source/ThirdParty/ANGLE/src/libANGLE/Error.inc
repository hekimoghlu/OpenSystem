/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 21, 2024.
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
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Error.inc: Inline definitions of egl::Error and gl::Error classes which encapsulate API errors
// and optional error messages.

#include "common/angleutils.h"

#include <cstdarg>

namespace egl
{

Error::Error(EGLint errorCode)
    : mCode(errorCode),
      mID(0)
{
}

Error::Error(const Error &other)
    : mCode(other.mCode),
      mID(other.mID)
{
    if (other.mMessage)
    {
        createMessageString();
        *mMessage = *(other.mMessage);
    }
}

Error::Error(Error &&other)
    : mCode(other.mCode),
      mID(other.mID),
      mMessage(std::move(other.mMessage))
{
}

Error &Error::operator=(const Error &other)
{
    mCode = other.mCode;
    mID = other.mID;

    if (other.mMessage)
    {
        createMessageString();
        *mMessage = *(other.mMessage);
    }
    else
    {
        mMessage.reset();
    }

    return *this;
}

Error &Error::operator=(Error &&other)
{
    if (this != &other)
    {
        mCode = other.mCode;
        mID = other.mID;
        mMessage = std::move(other.mMessage);
    }

    return *this;
}

EGLint Error::getCode() const
{
    return mCode;
}

EGLint Error::getID() const
{
    return mID;
}

bool Error::isError() const
{
    return (mCode != EGL_SUCCESS);
}

void Error::setCode(EGLint code)
{
    mCode = code;
}

// Static
Error Error::NoError()
{
    return Error(EGL_SUCCESS);
}

}
