/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 19, 2022.
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
//

// Error.cpp: Implements the egl::Error and gl::Error classes which encapsulate API errors
// and optional error messages.

#include "libANGLE/Error.h"

#include "common/angleutils.h"
#include "common/debug.h"
#include "common/utilities.h"

#include <cstdarg>

namespace
{
std::unique_ptr<std::string> EmplaceErrorString(std::string &&message)
{
    return message.empty() ? std::unique_ptr<std::string>()
                           : std::unique_ptr<std::string>(new std::string(std::move(message)));
}
}  // anonymous namespace

namespace egl
{

Error::Error(EGLint errorCode, std::string &&message)
    : mCode(errorCode), mID(errorCode), mMessage(EmplaceErrorString(std::move(message)))
{}

Error::Error(EGLint errorCode, EGLint id, std::string &&message)
    : mCode(errorCode), mID(id), mMessage(EmplaceErrorString(std::move(message)))
{}

void Error::createMessageString() const
{
    if (!mMessage)
    {
        mMessage.reset(new std::string(GetGenericErrorMessage(mCode)));
    }
}

const std::string &Error::getMessage() const
{
    createMessageString();
    return *mMessage;
}

std::ostream &operator<<(std::ostream &os, const Error &err)
{
    return gl::FmtHex(os, err.getCode());
}
}  // namespace egl

namespace angle
{
egl::Error ResultToEGL(Result result)
{
    if (result == Result::Continue)
        return egl::NoError();

    return egl::Error(EGL_BAD_ACCESS);
}
}  // namespace angle
