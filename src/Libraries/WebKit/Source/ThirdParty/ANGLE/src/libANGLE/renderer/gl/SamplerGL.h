/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 21, 2023.
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

// SamplerGL.h: Defines the rx::SamplerGL class, an implementation of SamplerImpl.

#ifndef LIBANGLE_RENDERER_GL_SAMPLERGL_H_
#define LIBANGLE_RENDERER_GL_SAMPLERGL_H_

#include "libANGLE/angletypes.h"
#include "libANGLE/renderer/SamplerImpl.h"

namespace rx
{

class FunctionsGL;
class StateManagerGL;

class SamplerGL : public SamplerImpl
{
  public:
    SamplerGL(const gl::SamplerState &state,
              const FunctionsGL *functions,
              StateManagerGL *stateManager);
    ~SamplerGL() override;

    angle::Result syncState(const gl::Context *context, const bool dirty) override;

    GLuint getSamplerID() const;

  private:
    const FunctionsGL *mFunctions;
    StateManagerGL *mStateManager;

    mutable gl::SamplerState mAppliedSamplerState;
    GLuint mSamplerID;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_SAMPLERGL_H_
