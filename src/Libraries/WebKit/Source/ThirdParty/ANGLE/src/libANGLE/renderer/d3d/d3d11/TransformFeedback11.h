/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 3, 2024.
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

// TransformFeedback11.h: Implements the abstract rx::TransformFeedbackImpl class.

#ifndef LIBANGLE_RENDERER_D3D_D3D11_TRANSFORMFEEDBACK11_H_
#define LIBANGLE_RENDERER_D3D_D3D11_TRANSFORMFEEDBACK11_H_

#include "common/platform.h"

#include "libANGLE/Error.h"
#include "libANGLE/angletypes.h"
#include "libANGLE/renderer/TransformFeedbackImpl.h"
#include "libANGLE/renderer/serial_utils.h"

namespace rx
{

class Renderer11;

class TransformFeedback11 : public TransformFeedbackImpl
{
  public:
    TransformFeedback11(const gl::TransformFeedbackState &state, Renderer11 *renderer);
    ~TransformFeedback11() override;

    angle::Result begin(const gl::Context *context, gl::PrimitiveMode primitiveMode) override;
    angle::Result end(const gl::Context *context) override;
    angle::Result pause(const gl::Context *context) override;
    angle::Result resume(const gl::Context *context) override;

    angle::Result bindIndexedBuffer(const gl::Context *context,
                                    size_t index,
                                    const gl::OffsetBindingPointer<gl::Buffer> &binding) override;

    void onApply();

    bool isDirty() const;

    UINT getNumSOBuffers() const;
    angle::Result getSOBuffers(const gl::Context *context,
                               const std::vector<ID3D11Buffer *> **buffersOut);
    const std::vector<UINT> &getSOBufferOffsets() const;

    UniqueSerial getSerial() const;

  private:
    Renderer11 *mRenderer;

    bool mIsDirty;
    std::vector<ID3D11Buffer *> mBuffers;
    std::vector<UINT> mBufferOffsets;

    UniqueSerial mSerial;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_D3D_D3D11_TRANSFORMFEEDBACK11_H_
