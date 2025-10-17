/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 21, 2024.
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
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// SamplerMtl.h:
//    Defines the class interface for SamplerMtl, implementing SamplerImpl.
//

#ifndef LIBANGLE_RENDERER_METAL_SAMPLERMTL_H_
#define LIBANGLE_RENDERER_METAL_SAMPLERMTL_H_

#include "libANGLE/renderer/SamplerImpl.h"
#include "libANGLE/renderer/metal/mtl_common.h"

namespace rx
{

class ContextMtl;

class SamplerMtl : public SamplerImpl
{
  public:
    SamplerMtl(const gl::SamplerState &state);
    ~SamplerMtl() override;

    void onDestroy(const gl::Context *context) override;
    angle::Result syncState(const gl::Context *context, const bool dirty) override;
    const mtl::AutoObjCPtr<id<MTLSamplerState>> &getSampler(ContextMtl *contextMtl);

  private:
    mtl::AutoObjCPtr<id<MTLSamplerState>> mSamplerState;

    // Cache compare mode & func to detect their changes and let ProgramMtl verify that
    // GL_TEXTURE_COMPARE_MODE is not GL_NONE on a shadow sampler.
    // TODO(http://anglebug.com/42263785): Once the validation code is implemented on front-end, it
    // is possible to remove these caching.
    GLenum mCompareMode = 0;
    GLenum mCompareFunc = 0;
};

}  // namespace rx

#endif
