/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 31, 2022.
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

// TransformFeedbackImpl.h: Defines the abstract rx::TransformFeedbackImpl class.

#ifndef LIBANGLE_RENDERER_TRANSFORMFEEDBACKIMPL_H_
#define LIBANGLE_RENDERER_TRANSFORMFEEDBACKIMPL_H_

#include "common/angleutils.h"
#include "libANGLE/TransformFeedback.h"

namespace rx
{

class TransformFeedbackImpl : angle::NonCopyable
{
  public:
    TransformFeedbackImpl(const gl::TransformFeedbackState &state) : mState(state) {}
    virtual ~TransformFeedbackImpl() {}
    virtual void onDestroy(const gl::Context *context) {}

    virtual angle::Result begin(const gl::Context *context, gl::PrimitiveMode primitiveMode) = 0;
    virtual angle::Result end(const gl::Context *context)                                    = 0;
    virtual angle::Result pause(const gl::Context *context)                                  = 0;
    virtual angle::Result resume(const gl::Context *context)                                 = 0;

    virtual angle::Result bindIndexedBuffer(
        const gl::Context *context,
        size_t index,
        const gl::OffsetBindingPointer<gl::Buffer> &binding) = 0;

    virtual angle::Result onLabelUpdate(const gl::Context *context);

  protected:
    const gl::TransformFeedbackState &mState;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_TRANSFORMFEEDBACKIMPL_H_
