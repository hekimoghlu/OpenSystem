/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 11, 2022.
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

// egl_utils.h: Utility routines specific to the EGL->EGL implementation.

#ifndef LIBANGLE_RENDERER_GL_EGL_EGLUTILS_H_
#define LIBANGLE_RENDERER_GL_EGL_EGLUTILS_H_

#include <vector>

#include "common/platform.h"
#include "libANGLE/AttributeMap.h"

namespace rx
{

namespace native_egl
{

using AttributeVector = std::vector<EGLint>;

// Filter the attribute map and return a vector of attributes that can be passed to the native
// driver.  Does NOT append EGL_NONE to the vector.
AttributeVector TrimAttributeMap(const egl::AttributeMap &attributes,
                                 const EGLint *forwardAttribs,
                                 size_t forwardAttribsCount);

template <size_t N>
AttributeVector TrimAttributeMap(const egl::AttributeMap &attributes,
                                 const EGLint (&forwardAttribs)[N])
{
    return TrimAttributeMap(attributes, forwardAttribs, N);
}

// Append EGL_NONE to the attribute vector so that it can be passed to a native driver.
void FinalizeAttributeVector(AttributeVector *attributeVector);

}  // namespace native_egl

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_EGL_EGLUTILS_H_
