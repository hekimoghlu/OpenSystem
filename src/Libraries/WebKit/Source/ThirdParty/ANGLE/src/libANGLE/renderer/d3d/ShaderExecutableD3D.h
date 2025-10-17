/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 30, 2023.
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
// Copyright 2012 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// ShaderExecutable.h: Defines a class to contain D3D shader executable
// implementation details.

#ifndef LIBANGLE_RENDERER_D3D_SHADEREXECUTABLED3D_H_
#define LIBANGLE_RENDERER_D3D_SHADEREXECUTABLED3D_H_

#include "common/MemoryBuffer.h"
#include "common/debug.h"

#include <cstdint>
#include <vector>

namespace rx
{

class ShaderExecutableD3D : angle::NonCopyable
{
  public:
    ShaderExecutableD3D(const void *function, size_t length);
    virtual ~ShaderExecutableD3D();

    const uint8_t *getFunction() const;

    size_t getLength() const;

    const std::string &getDebugInfo() const;

    void appendDebugInfo(const std::string &info);

  private:
    std::vector<uint8_t> mFunctionBuffer;
    std::string mDebugInfo;
};

class UniformStorageD3D : angle::NonCopyable
{
  public:
    UniformStorageD3D(size_t initialSize);
    virtual ~UniformStorageD3D();

    size_t size() const;

    uint8_t *getDataPointer(unsigned int registerIndex, unsigned int registerElement);

  private:
    angle::MemoryBuffer mUniformData;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_D3D_SHADEREXECUTABLED3D_H_
