/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 15, 2024.
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
// TransformFeedbackNULL.cpp:
//    Implements the class methods for TransformFeedbackNULL.
//

#include "libANGLE/renderer/null/TransformFeedbackNULL.h"

#include "common/debug.h"

namespace rx
{

TransformFeedbackNULL::TransformFeedbackNULL(const gl::TransformFeedbackState &state)
    : TransformFeedbackImpl(state)
{}

TransformFeedbackNULL::~TransformFeedbackNULL() {}

angle::Result TransformFeedbackNULL::begin(const gl::Context *context,
                                           gl::PrimitiveMode primitiveMode)
{
    return angle::Result::Continue;
}

angle::Result TransformFeedbackNULL::end(const gl::Context *context)
{
    return angle::Result::Continue;
}

angle::Result TransformFeedbackNULL::pause(const gl::Context *context)
{
    return angle::Result::Continue;
}

angle::Result TransformFeedbackNULL::resume(const gl::Context *context)
{
    return angle::Result::Continue;
}

angle::Result TransformFeedbackNULL::bindIndexedBuffer(
    const gl::Context *context,
    size_t index,
    const gl::OffsetBindingPointer<gl::Buffer> &binding)
{
    return angle::Result::Continue;
}

}  // namespace rx
