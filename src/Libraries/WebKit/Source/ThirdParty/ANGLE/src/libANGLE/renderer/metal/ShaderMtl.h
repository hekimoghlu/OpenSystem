/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 25, 2025.
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
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ShaderMtl.h:
//    Defines the class interface for ShaderMtl, implementing ShaderImpl.
//
#ifndef LIBANGLE_RENDERER_METAL_SHADERMTL_H_
#define LIBANGLE_RENDERER_METAL_SHADERMTL_H_

#include <map>

#include "libANGLE/renderer/ShaderImpl.h"
#include "libANGLE/renderer/metal/mtl_msl_utils.h"

namespace rx
{
class ShaderMtl : public ShaderImpl
{
  public:
    ShaderMtl(const gl::ShaderState &state);
    ~ShaderMtl() override;

    std::shared_ptr<ShaderTranslateTask> compile(const gl::Context *context,
                                                 ShCompileOptions *options) override;
    std::shared_ptr<ShaderTranslateTask> load(const gl::Context *context,
                                              gl::BinaryInputStream *stream) override;

    const SharedCompiledShaderStateMtl &getCompiledState() const { return mCompiledState; }

    std::string getDebugInfo() const override;

  private:
    SharedCompiledShaderStateMtl mCompiledState;
};

}  // namespace rx

#endif /* LIBANGLE_RENDERER_METAL_SHADERMTL_H_ */
