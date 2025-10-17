/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 19, 2022.
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
// MemoryShaderCache: Stores compiled shader in memory so they don't
//   always have to be re-compiled. Can be used in conjunction with the platform
//   layer to warm up the cache from disk.

#include "libANGLE/MemoryShaderCache.h"

#include <GLSLANG/ShaderVars.h>
#include <anglebase/sha1.h>

#include "common/BinaryStream.h"
#include "common/utilities.h"
#include "libANGLE/Compiler.h"
#include "libANGLE/Context.h"
#include "libANGLE/Debug.h"
#include "libANGLE/Uniform.h"
#include "libANGLE/histogram_macros.h"
#include "libANGLE/renderer/ShaderImpl.h"
#include "platform/PlatformMethods.h"

namespace gl
{

namespace
{
// Limit decompressed programs to 5MB. If they're larger then this there is a good chance the data
// is not what we expect. This limits the amount of memory we will allocate based on a binary blob
// we believe is compressed data.
static constexpr size_t kMaxUncompressedShaderSize = 5 * 1024 * 1024;
}  // namespace

MemoryShaderCache::MemoryShaderCache(egl::BlobCache &blobCache) : mBlobCache(blobCache) {}

MemoryShaderCache::~MemoryShaderCache() {}

egl::CacheGetResult MemoryShaderCache::getShader(const Context *context,
                                                 Shader *shader,
                                                 const egl::BlobCache::Key &shaderHash,
                                                 angle::JobResultExpectancy resultExpectancy)
{
    // If caching is effectively disabled, don't bother calculating the hash.
    if (!mBlobCache.isCachingEnabled(context))
    {
        return egl::CacheGetResult::NotFound;
    }

    angle::MemoryBuffer uncompressedData;
    const egl::BlobCache::GetAndDecompressResult result =
        mBlobCache.getAndDecompress(context, context->getScratchBuffer(), shaderHash,
                                    kMaxUncompressedShaderSize, &uncompressedData);
    switch (result)
    {
        case egl::BlobCache::GetAndDecompressResult::DecompressFailure:
            ANGLE_PERF_WARNING(context->getState().getDebug(), GL_DEBUG_SEVERITY_LOW,
                               "Error decompressing shader binary data from cache.");
            mBlobCache.remove(shaderHash);
            return egl::CacheGetResult::NotFound;

        case egl::BlobCache::GetAndDecompressResult::NotFound:
            return egl::CacheGetResult::NotFound;

        case egl::BlobCache::GetAndDecompressResult::Success:
            if (shader->loadBinary(context, uncompressedData.data(),
                                   static_cast<int>(uncompressedData.size()), resultExpectancy))
            {
                return egl::CacheGetResult::Success;
            }

            // Cache load failed, evict.
            ANGLE_PERF_WARNING(context->getState().getDebug(), GL_DEBUG_SEVERITY_LOW,
                               "Failed to load shader binary from cache.");
            mBlobCache.remove(shaderHash);
            return egl::CacheGetResult::Rejected;
    }

    UNREACHABLE();
    return egl::CacheGetResult::NotFound;
}

angle::Result MemoryShaderCache::putShader(const Context *context,
                                           const egl::BlobCache::Key &shaderHash,
                                           const Shader *shader)
{
    // If caching is effectively disabled, don't bother serializing the shader.
    if (!mBlobCache.isCachingEnabled(context))
    {
        return angle::Result::Continue;
    }

    angle::MemoryBuffer serializedShader;
    ANGLE_TRY(shader->serialize(nullptr, &serializedShader));

    size_t compressedSize;
    if (!mBlobCache.compressAndPut(context, shaderHash, std::move(serializedShader),
                                   &compressedSize))
    {
        ANGLE_PERF_WARNING(context->getState().getDebug(), GL_DEBUG_SEVERITY_LOW,
                           "Error compressing shader binary data for insertion into cache.");
        return angle::Result::Continue;
    }

    return angle::Result::Continue;
}

void MemoryShaderCache::clear()
{
    mBlobCache.clear();
}

size_t MemoryShaderCache::maxSize() const
{
    return mBlobCache.maxSize();
}

}  // namespace gl
