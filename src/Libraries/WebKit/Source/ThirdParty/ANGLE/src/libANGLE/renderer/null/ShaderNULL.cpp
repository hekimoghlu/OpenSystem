/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 8, 2025.
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
// ShaderNULL.cpp:
//    Implements the class methods for ShaderNULL.
//

#include "libANGLE/renderer/null/ShaderNULL.h"

#include "common/debug.h"
#include "libANGLE/Context.h"
#include "libANGLE/renderer/ContextImpl.h"

namespace rx
{

ShaderNULL::ShaderNULL(const gl::ShaderState &data) : ShaderImpl(data) {}

ShaderNULL::~ShaderNULL() {}

std::shared_ptr<ShaderTranslateTask> ShaderNULL::compile(const gl::Context *context,
                                                         ShCompileOptions *options)
{
    const gl::Extensions &extensions = context->getImplementation()->getExtensions();
    if (extensions.shaderPixelLocalStorageANGLE)
    {
        options->pls = context->getImplementation()->getNativePixelLocalStorageOptions();
    }
    return std::shared_ptr<ShaderTranslateTask>(new ShaderTranslateTask);
}

std::shared_ptr<ShaderTranslateTask> ShaderNULL::load(const gl::Context *context,
                                                      gl::BinaryInputStream *stream)
{
    return std::shared_ptr<ShaderTranslateTask>(new ShaderTranslateTask);
}

std::string ShaderNULL::getDebugInfo() const
{
    return "";
}

}  // namespace rx
