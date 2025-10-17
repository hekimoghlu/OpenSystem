/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 11, 2023.
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

// SamplerImpl.h: Defines the abstract rx::SamplerImpl class.

#ifndef LIBANGLE_RENDERER_SAMPLERIMPL_H_
#define LIBANGLE_RENDERER_SAMPLERIMPL_H_

#include "common/angleutils.h"
#include "libANGLE/Error.h"
#include "libANGLE/Sampler.h"

namespace gl
{
class Context;
class SamplerState;
}  // namespace gl

namespace rx
{

class SamplerImpl : angle::NonCopyable
{
  public:
    SamplerImpl(const gl::SamplerState &state) : mState(state) {}
    virtual ~SamplerImpl() {}

    virtual void onDestroy(const gl::Context *context)
    {
        // Default implementation: no-op.
    }
    virtual angle::Result syncState(const gl::Context *context, const bool dirty) = 0;

    angle::Result onLabelUpdate(const gl::Context *context) { return angle::Result::Continue; }

  protected:
    const gl::SamplerState &mState;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_SAMPLERIMPL_H_
