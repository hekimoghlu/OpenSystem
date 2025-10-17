/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 10, 2022.
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
// Copyright 2023 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// mtl_pipeline_cache.h:
//    Defines classes for caching of mtl pipelines
//

#ifndef LIBANGLE_RENDERER_METAL_MTL_PIPELINE_CACHE_H_
#define LIBANGLE_RENDERER_METAL_MTL_PIPELINE_CACHE_H_

#include "common/hash_utils.h"
#include "libANGLE/SizedMRUCache.h"
#include "libANGLE/renderer/metal/mtl_utils.h"

namespace rx
{
namespace mtl
{

struct PipelineKey
{
    AutoObjCPtr<id<MTLFunction>> vertexShader;
    AutoObjCPtr<id<MTLFunction>> fragmentShader;
    RenderPipelineDesc pipelineDesc;

    AutoObjCPtr<id<MTLFunction>> computeShader;

    bool isRenderPipeline() const;

    bool operator==(const PipelineKey &rhs) const;
    size_t hash() const;
};

}  // namespace mtl
}  // namespace rx

namespace std
{

template <>
struct hash<rx::mtl::PipelineKey>
{
    size_t operator()(const rx::mtl::PipelineKey &key) const { return key.hash(); }
};

}  // namespace std

namespace rx
{
namespace mtl
{

class PipelineCache : angle::NonCopyable
{
  public:
    PipelineCache();

    angle::Result getRenderPipeline(ContextMtl *context,
                                    id<MTLFunction> vertexShader,
                                    id<MTLFunction> fragmentShader,
                                    const RenderPipelineDesc &desc,
                                    AutoObjCPtr<id<MTLRenderPipelineState>> *outRenderPipeline);
    angle::Result getComputePipeline(ContextMtl *context,
                                     id<MTLFunction> computeShader,
                                     AutoObjCPtr<id<MTLComputePipelineState>> *outComputePipeline);

  private:
    static constexpr unsigned int kMaxPipelines = 128;

    // The cache tries to clean up this many states at once.
    static constexpr unsigned int kGCLimit = 32;

    struct PipelineVariant
    {
        AutoObjCPtr<id<MTLRenderPipelineState>> renderPipeline;
        AutoObjCPtr<id<MTLComputePipelineState>> computePipeline;
    };

    using RenderPipelineMap = angle::base::HashingMRUCache<PipelineKey, PipelineVariant>;
    RenderPipelineMap mPipelineCache;
};

}  // namespace mtl
}  // namespace rx

#endif  // LIBANGLE_RENDERER_METAL_MTL_PIPELINE_CACHE_H_
