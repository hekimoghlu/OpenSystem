/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 24, 2025.
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
// Copyright 2024 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ShaderWgpu.cpp:
//    Implements the class methods for ShaderWgpu.
//

#include "libANGLE/renderer/wgpu/ShaderWgpu.h"

#include "common/debug.h"
#include "libANGLE/Context.h"
#include "libANGLE/renderer/ContextImpl.h"
#include "libANGLE/trace.h"

namespace rx
{

namespace
{
class ShaderTranslateTaskWgpu final : public ShaderTranslateTask
{
    bool translate(ShHandle compiler,
                   const ShCompileOptions &options,
                   const std::string &source) override
    {
        ANGLE_TRACE_EVENT1("gpu.angle", "ShaderTranslateTaskWgpu::translate", "source", source);

        const char *srcStrings[] = {source.c_str()};
        return sh::Compile(compiler, srcStrings, ArraySize(srcStrings), options);
    }

    void postTranslate(ShHandle compiler, const gl::CompiledShaderState &compiledState) override {}
};
}  // namespace

ShaderWgpu::ShaderWgpu(const gl::ShaderState &data) : ShaderImpl(data) {}

ShaderWgpu::~ShaderWgpu() {}

std::shared_ptr<ShaderTranslateTask> ShaderWgpu::compile(const gl::Context *context,
                                                         ShCompileOptions *options)
{
    const gl::Extensions &extensions = context->getImplementation()->getExtensions();
    if (extensions.shaderPixelLocalStorageANGLE)
    {
        options->pls = context->getImplementation()->getNativePixelLocalStorageOptions();
    }

    options->validateAST = true;

    options->separateCompoundStructDeclarations = true;

    return std::shared_ptr<ShaderTranslateTask>(new ShaderTranslateTaskWgpu);
}

std::shared_ptr<ShaderTranslateTask> ShaderWgpu::load(const gl::Context *context,
                                                      gl::BinaryInputStream *stream)
{
    UNREACHABLE();
    return std::shared_ptr<ShaderTranslateTask>(new ShaderTranslateTask);
}

std::string ShaderWgpu::getDebugInfo() const
{
    return "";
}

}  // namespace rx
