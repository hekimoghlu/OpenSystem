/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 4, 2023.
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
// Copyright 2021 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RenderDoc:
//   Connection to renderdoc for capturing tests through its API.
//

#include "RenderDoc.h"

#include "common/angleutils.h"
#include "common/debug.h"

RenderDoc::RenderDoc() : mRenderDocModule(nullptr), mApi(nullptr) {}

RenderDoc::~RenderDoc()
{
    SafeDelete(mRenderDocModule);
}

#if defined(ANGLE_PLATFORM_ANDROID) || defined(ANGLE_PLATFORM_LINUX) || \
    defined(ANGLE_PLATFORM_WINDOWS)
#    include "third_party/renderdoc/src/renderdoc_app.h"

#    if defined(ANGLE_PLATFORM_WINDOWS)
constexpr char kRenderDocModuleName[] = "renderdoc";
#    elif defined(ANGLE_PLATFORM_ANDROID)
constexpr char kRenderDocModuleName[] = "libVkLayer_GLES_RenderDoc";
#    else
constexpr char kRenderDocModuleName[] = "librenderdoc";
#    endif

void RenderDoc::attach()
{
    mRenderDocModule = OpenSharedLibrary(kRenderDocModuleName, angle::SearchType::AlreadyLoaded);
    if (mRenderDocModule == nullptr || mRenderDocModule->getNative() == nullptr)
    {
        return;
    }
    void *getApi = mRenderDocModule->getSymbol("RENDERDOC_GetAPI");
    if (getApi == nullptr)
    {
        return;
    }

    int result = reinterpret_cast<pRENDERDOC_GetAPI>(getApi)(eRENDERDOC_API_Version_1_1_2, &mApi);
    if (result != 1)
    {
        ERR() << "RenderDoc module is present but API 1.1.2 is unavailable";
        mApi = nullptr;
    }
}

void RenderDoc::startFrame()
{
    if (mApi)
    {
        static_cast<RENDERDOC_API_1_1_2 *>(mApi)->StartFrameCapture(nullptr, nullptr);
    }
}

void RenderDoc::endFrame()
{
    if (mApi)
    {
        static_cast<RENDERDOC_API_1_1_2 *>(mApi)->EndFrameCapture(nullptr, nullptr);
    }
}

#else  // defiend(ANGLE_PLATFORM_ANDROID) || defined(ANGLE_PLATFORM_LINUX) ||
       // defined(ANGLE_PLATFORM_WINDOWS)

// Stub out the implementation on unsupported platforms.
void RenderDoc::attach()
{
    mApi = nullptr;
}

void RenderDoc::startFrame() {}

void RenderDoc::endFrame() {}

#endif  // defiend(ANGLE_PLATFORM_ANDROID) || defined(ANGLE_PLATFORM_LINUX) ||
        // defined(ANGLE_PLATFORM_WINDOWS)
