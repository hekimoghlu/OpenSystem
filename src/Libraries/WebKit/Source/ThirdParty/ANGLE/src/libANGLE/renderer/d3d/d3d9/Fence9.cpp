/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 19, 2023.
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
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Fence9.cpp: Defines the rx::FenceNV9 class.

#include "libANGLE/renderer/d3d/d3d9/Fence9.h"

#include "libANGLE/Context.h"
#include "libANGLE/renderer/d3d/d3d9/Context9.h"
#include "libANGLE/renderer/d3d/d3d9/Renderer9.h"
#include "libANGLE/renderer/d3d/d3d9/renderer9_utils.h"

namespace rx
{

FenceNV9::FenceNV9(Renderer9 *renderer) : FenceNVImpl(), mRenderer(renderer), mQuery(nullptr) {}

FenceNV9::~FenceNV9()
{
    SafeRelease(mQuery);
}

angle::Result FenceNV9::set(const gl::Context *context, GLenum condition)
{
    if (!mQuery)
    {
        ANGLE_TRY(mRenderer->allocateEventQuery(context, &mQuery));
    }

    HRESULT result = mQuery->Issue(D3DISSUE_END);
    if (FAILED(result))
    {
        SafeRelease(mQuery);
    }
    ANGLE_TRY_HR(GetImplAs<Context9>(context), result, "Failed to end event query");
    return angle::Result::Continue;
}

angle::Result FenceNV9::test(const gl::Context *context, GLboolean *outFinished)
{
    return testHelper(GetImplAs<Context9>(context), true, outFinished);
}

angle::Result FenceNV9::finish(const gl::Context *context)
{
    GLboolean finished = GL_FALSE;
    while (finished != GL_TRUE)
    {
        ANGLE_TRY(testHelper(GetImplAs<Context9>(context), true, &finished));
        Sleep(0);
    }

    return angle::Result::Continue;
}

angle::Result FenceNV9::testHelper(Context9 *context9,
                                   bool flushCommandBuffer,
                                   GLboolean *outFinished)
{
    ASSERT(mQuery);

    DWORD getDataFlags = (flushCommandBuffer ? D3DGETDATA_FLUSH : 0);
    HRESULT result     = mQuery->GetData(nullptr, 0, getDataFlags);
    ANGLE_TRY_HR(context9, result, "Failed to get query data");
    ASSERT(result == S_OK || result == S_FALSE);
    *outFinished = ((result == S_OK) ? GL_TRUE : GL_FALSE);
    return angle::Result::Continue;
}

}  // namespace rx
