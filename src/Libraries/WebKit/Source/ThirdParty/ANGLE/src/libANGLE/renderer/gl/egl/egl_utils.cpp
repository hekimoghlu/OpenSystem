/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 2, 2024.
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
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// egl_utils.cpp: Utility routines specific to the EGL->EGL implementation.

#include "libANGLE/renderer/gl/egl/egl_utils.h"

#include "common/debug.h"

namespace rx
{

namespace native_egl
{

AttributeVector TrimAttributeMap(const egl::AttributeMap &attributes,
                                 const EGLint *forwardAttribs,
                                 size_t forwardAttribsCount)
{
    AttributeVector result;
    for (size_t forwardAttribIndex = 0; forwardAttribIndex < forwardAttribsCount;
         forwardAttribIndex++)
    {
        EGLint forwardAttrib = forwardAttribs[forwardAttribIndex];
        if (attributes.contains(forwardAttrib))
        {
            result.push_back(forwardAttrib);
            result.push_back(static_cast<int>(attributes.get(forwardAttrib)));
        }
    }
    return result;
}

void FinalizeAttributeVector(AttributeVector *attributeVector)
{
    ASSERT(attributeVector);
    attributeVector->push_back(EGL_NONE);
}

}  // namespace native_egl

}  // namespace rx
