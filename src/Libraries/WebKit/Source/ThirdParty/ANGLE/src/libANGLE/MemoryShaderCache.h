/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 16, 2025.
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
// Copyright 2022 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// MemoryShaderCache: Stores compiled shaders in memory so they don't
//   always have to be re-compiled. Can be used in conjunction with the platform
//   layer to warm up the cache from disk.

#ifndef LIBANGLE_MEMORY_SHADER_CACHE_H_
#define LIBANGLE_MEMORY_SHADER_CACHE_H_

#include <array>

#include "GLSLANG/ShaderLang.h"
#include "common/MemoryBuffer.h"
#include "libANGLE/BlobCache.h"
#include "libANGLE/Error.h"

namespace gl
{
class Context;
class Shader;
class ShaderState;
class ShCompilerInstance;

class MemoryShaderCache final : angle::NonCopyable
{
  public:
    explicit MemoryShaderCache(egl::BlobCache &blobCache);
    ~MemoryShaderCache();

    // Helper method that serializes a shader.
    angle::Result putShader(const Context *context,
                            const egl::BlobCache::Key &shaderHash,
                            const Shader *shader);

    // Check the cache, and deserialize and load the shader if found. Evict existing hash if load
    // fails.
    egl::CacheGetResult getShader(const Context *context,
                                  Shader *shader,
                                  const egl::BlobCache::Key &shaderHash,
                                  angle::JobResultExpectancy resultExpectancy);

    // Empty the cache.
    void clear();

    // Returns the maximum cache size in bytes.
    size_t maxSize() const;

  private:
    egl::BlobCache &mBlobCache;
};

}  // namespace gl

#endif  // LIBANGLE_MEMORY_SHADER_CACHE_H_
