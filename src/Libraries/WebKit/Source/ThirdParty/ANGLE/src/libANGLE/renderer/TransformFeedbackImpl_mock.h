/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 11, 2025.
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
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// TransformFeedbackImpl_mock.h: Defines a mock of the TransformFeedbackImpl class.

#ifndef LIBANGLE_RENDERER_TRANSFORMFEEDBACKIMPLMOCK_H_
#define LIBANGLE_RENDERER_TRANSFORMFEEDBACKIMPLMOCK_H_

#include "gmock/gmock.h"

#include "libANGLE/renderer/TransformFeedbackImpl.h"

namespace rx
{

class MockTransformFeedbackImpl : public TransformFeedbackImpl
{
  public:
    MockTransformFeedbackImpl(const gl::TransformFeedbackState &state)
        : TransformFeedbackImpl(state)
    {}
    ~MockTransformFeedbackImpl() { destructor(); }

    MOCK_METHOD2(begin, angle::Result(const gl::Context *, gl::PrimitiveMode));
    MOCK_METHOD1(end, angle::Result(const gl::Context *));
    MOCK_METHOD1(pause, angle::Result(const gl::Context *));
    MOCK_METHOD1(resume, angle::Result(const gl::Context *));

    MOCK_METHOD3(bindIndexedBuffer,
                 angle::Result(const gl::Context *,
                               size_t,
                               const gl::OffsetBindingPointer<gl::Buffer> &));

    MOCK_METHOD0(destructor, void());
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_TRANSFORMFEEDBACKIMPLMOCK_H_
