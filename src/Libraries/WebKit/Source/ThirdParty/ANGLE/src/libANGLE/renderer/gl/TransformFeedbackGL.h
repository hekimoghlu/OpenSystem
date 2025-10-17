/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 13, 2023.
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

// TransformFeedbackGL.h: Defines the class interface for TransformFeedbackGL.

#ifndef LIBANGLE_RENDERER_GL_TRANSFORMFEEDBACKGL_H_
#define LIBANGLE_RENDERER_GL_TRANSFORMFEEDBACKGL_H_

#include "libANGLE/renderer/TransformFeedbackImpl.h"

namespace rx
{

class FunctionsGL;
class StateManagerGL;

class TransformFeedbackGL : public TransformFeedbackImpl
{
  public:
    TransformFeedbackGL(const gl::TransformFeedbackState &state,
                        const FunctionsGL *functions,
                        StateManagerGL *stateManager);
    ~TransformFeedbackGL() override;

    angle::Result begin(const gl::Context *context, gl::PrimitiveMode primitiveMode) override;
    angle::Result end(const gl::Context *context) override;
    angle::Result pause(const gl::Context *context) override;
    angle::Result resume(const gl::Context *context) override;

    angle::Result bindIndexedBuffer(const gl::Context *context,
                                    size_t index,
                                    const gl::OffsetBindingPointer<gl::Buffer> &binding) override;

    GLuint getTransformFeedbackID() const;

    void syncActiveState(const gl::Context *context,
                         bool active,
                         gl::PrimitiveMode primitiveMode) const;
    void syncPausedState(bool paused) const;

  private:
    const FunctionsGL *mFunctions;
    StateManagerGL *mStateManager;

    GLuint mTransformFeedbackID;

    mutable bool mIsActive;
    mutable bool mIsPaused;
    mutable GLuint mActiveProgram;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_TRANSFORMFEEDBACKGL_H_
