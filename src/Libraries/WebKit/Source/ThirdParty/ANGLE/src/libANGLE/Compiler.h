/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 7, 2024.
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

// Compiler.h: Defines the gl::Compiler class, abstracting the ESSL compiler
// that a GL context holds.

#ifndef LIBANGLE_COMPILER_H_
#define LIBANGLE_COMPILER_H_

#include <vector>

#include "GLSLANG/ShaderLang.h"
#include "common/PackedEnums.h"
#include "libANGLE/Error.h"
#include "libANGLE/RefCountObject.h"

namespace rx
{
class CompilerImpl;
class GLImplFactory;
}  // namespace rx

namespace gl
{
class ShCompilerInstance;
class State;

class Compiler final : public RefCountObjectNoID
{
  public:
    Compiler(rx::GLImplFactory *implFactory, const State &data, egl::Display *display);

    void onDestroy(const Context *context) override;

    ShCompilerInstance getInstance(ShaderType shaderType);
    void putInstance(ShCompilerInstance &&instance);
    ShShaderOutput getShaderOutputType() const { return mOutputType; }
    const ShBuiltInResources &getBuiltInResources() const { return mResources; }

    static ShShaderSpec SelectShaderSpec(const State &state);

  private:
    ~Compiler() override;
    std::unique_ptr<rx::CompilerImpl> mImplementation;
    ShShaderSpec mSpec;
    ShShaderOutput mOutputType;
    ShBuiltInResources mResources;
    ShaderMap<std::vector<ShCompilerInstance>> mPools;
};

class ShCompilerInstance final : public angle::NonCopyable
{
  public:
    ShCompilerInstance();
    ShCompilerInstance(ShHandle handle, ShShaderOutput outputType, ShaderType shaderType);
    ~ShCompilerInstance();
    void destroy();

    ShCompilerInstance(ShCompilerInstance &&other);
    ShCompilerInstance &operator=(ShCompilerInstance &&other);

    ShHandle getHandle();
    ShaderType getShaderType() const;
    ShBuiltInResources getBuiltInResources() const;
    ShShaderOutput getShaderOutputType() const;

  private:
    ShHandle mHandle;
    ShShaderOutput mOutputType;
    ShaderType mShaderType;
};

}  // namespace gl

#endif  // LIBANGLE_COMPILER_H_
