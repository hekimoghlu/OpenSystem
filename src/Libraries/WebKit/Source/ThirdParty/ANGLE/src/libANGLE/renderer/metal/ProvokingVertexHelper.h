/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 12, 2023.
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
// Copyright 2021 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ProvokingVertexHelper.h:
//    Manages re-writing index buffers when provoking vertex support is needed.
//    Handles points, lines, triangles, line strips, and triangle strips.
//    Line loops, and triangle fans should be pre-processed.
//
#ifndef LIBANGLE_RENDERER_METAL_PROVOKINGVERTEXHELPER_H
#define LIBANGLE_RENDERER_METAL_PROVOKINGVERTEXHELPER_H

#include "libANGLE/Context.h"
#include "libANGLE/renderer/ContextImpl.h"
#include "libANGLE/renderer/metal/DisplayMtl.h"
#include "libANGLE/renderer/metal/mtl_buffer_pool.h"
#include "libANGLE/renderer/metal/mtl_command_buffer.h"
#include "libANGLE/renderer/metal/mtl_context_device.h"
#include "libANGLE/renderer/metal/mtl_state_cache.h"
namespace rx
{
class ContextMtl;

class ProvokingVertexHelper : angle::NonCopyable
{
  public:
    ProvokingVertexHelper(ContextMtl *context);
    angle::Result preconditionIndexBuffer(ContextMtl *context,
                                          mtl::BufferRef indexBuffer,
                                          size_t indexCount,
                                          size_t indexOffset,
                                          bool primitiveRestartEnabled,
                                          gl::PrimitiveMode primitiveMode,
                                          gl::DrawElementsType elementsType,
                                          size_t &outIndexCount,
                                          size_t &outIndexOffset,
                                          gl::PrimitiveMode &outPrimitiveMode,
                                          mtl::BufferRef &outNewBuffer);

    angle::Result generateIndexBuffer(ContextMtl *context,
                                      size_t first,
                                      size_t indexCount,
                                      gl::PrimitiveMode primitiveMode,
                                      gl::DrawElementsType elementsType,
                                      size_t &outIndexCount,
                                      size_t &outIndexOffset,
                                      gl::PrimitiveMode &outPrimitiveMode,
                                      mtl::BufferRef &outNewBuffer);

    void releaseInFlightBuffers(ContextMtl *contextMtl);
    void ensureCommandBufferReady();
    void onDestroy(ContextMtl *context);
    mtl::ComputeCommandEncoder *getComputeCommandEncoder();

  private:
    angle::Result getComputePipleineState(
        ContextMtl *context,
        const mtl::ProvokingVertexComputePipelineDesc &desc,
        mtl::AutoObjCPtr<id<MTLComputePipelineState>> *outComputePipeline);

    angle::Result prepareCommandEncoderForDescriptor(ContextMtl *context,
                                                     mtl::ComputeCommandEncoder *encoder,
                                                     mtl::ProvokingVertexComputePipelineDesc desc);

    mtl::BufferPool mIndexBuffers;

    std::unordered_map<mtl::ProvokingVertexComputePipelineDesc, mtl::AutoObjCPtr<id<MTLFunction>>>
        mComputeFunctions;
};
}  // namespace rx
#endif /* LIBANGLE_RENDERER_METAL_PROVOKINGVERTEXHELPER_H */
