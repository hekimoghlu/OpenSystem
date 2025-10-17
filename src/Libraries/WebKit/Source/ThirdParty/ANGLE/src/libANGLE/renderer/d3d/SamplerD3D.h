/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 10, 2024.
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

// SamplerD3D.h: Defines the rx::SamplerD3D class, an implementation of SamplerImpl.

#ifndef LIBANGLE_RENDERER_D3D_SAMPLERD3D_H_
#define LIBANGLE_RENDERER_D3D_SAMPLERD3D_H_

#include "libANGLE/renderer/SamplerImpl.h"

namespace rx
{

class SamplerD3D : public SamplerImpl
{
  public:
    SamplerD3D(const gl::SamplerState &state) : SamplerImpl(state) {}
    ~SamplerD3D() override {}

    angle::Result syncState(const gl::Context *context, const bool dirtyBits) override;
};

inline angle::Result SamplerD3D::syncState(const gl::Context *context, const bool dirtyBits)
{
    return angle::Result::Continue;
}

}  // namespace rx

#endif  // LIBANGLE_RENDERER_D3D_SAMPLERD3D_H_
