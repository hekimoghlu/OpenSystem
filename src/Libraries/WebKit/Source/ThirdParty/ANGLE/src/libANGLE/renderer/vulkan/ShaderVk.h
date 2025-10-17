/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 15, 2025.
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
// ShaderVk.h:
//    Defines the class interface for ShaderVk, implementing ShaderImpl.
//

#ifndef LIBANGLE_RENDERER_VULKAN_SHADERVK_H_
#define LIBANGLE_RENDERER_VULKAN_SHADERVK_H_

#include "libANGLE/renderer/ShaderImpl.h"

namespace rx
{

class ShaderVk : public ShaderImpl
{
  public:
    ShaderVk(const gl::ShaderState &state);
    ~ShaderVk() override;

    std::shared_ptr<ShaderTranslateTask> compile(const gl::Context *context,
                                                 ShCompileOptions *options) override;
    std::shared_ptr<ShaderTranslateTask> load(const gl::Context *context,
                                              gl::BinaryInputStream *stream) override;

    std::string getDebugInfo() const override;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_VULKAN_SHADERVK_H_
