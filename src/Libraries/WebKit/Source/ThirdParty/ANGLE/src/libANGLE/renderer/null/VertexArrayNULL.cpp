/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 11, 2021.
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
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// VertexArrayNULL.cpp:
//    Implements the class methods for VertexArrayNULL.
//

#include "libANGLE/renderer/null/VertexArrayNULL.h"

#include "common/debug.h"

namespace rx
{

VertexArrayNULL::VertexArrayNULL(const gl::VertexArrayState &data) : VertexArrayImpl(data) {}

angle::Result VertexArrayNULL::syncState(const gl::Context *context,
                                         const gl::VertexArray::DirtyBits &dirtyBits,
                                         gl::VertexArray::DirtyAttribBitsArray *attribBits,
                                         gl::VertexArray::DirtyBindingBitsArray *bindingBits)
{
    // Clear the dirty bits in the back-end here.
    memset(attribBits, 0, sizeof(gl::VertexArray::DirtyAttribBitsArray));
    memset(bindingBits, 0, sizeof(gl::VertexArray::DirtyBindingBitsArray));

    return angle::Result::Continue;
}

}  // namespace rx
