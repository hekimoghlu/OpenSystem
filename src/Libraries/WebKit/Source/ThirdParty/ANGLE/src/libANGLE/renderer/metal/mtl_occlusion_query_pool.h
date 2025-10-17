/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 25, 2023.
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
// Copyright (c) 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// mtl_occlusion_query_pool: A pool for allocating visibility query within
// one render pass.
//

#ifndef LIBANGLE_RENDERER_METAL_MTL_OCCLUSION_QUERY_POOL_H_
#define LIBANGLE_RENDERER_METAL_MTL_OCCLUSION_QUERY_POOL_H_

#include <vector>

#include "libANGLE/Context.h"
#include "libANGLE/renderer/metal/mtl_common.h"
#include "libANGLE/renderer/metal/mtl_resources.h"

namespace rx
{

class ContextMtl;
class QueryMtl;

namespace mtl
{

class OcclusionQueryPool
{
  public:
    OcclusionQueryPool();
    ~OcclusionQueryPool();

    void destroy(ContextMtl *contextMtl);

    // Allocate an offset in visibility buffer for a query in a render pass.
    // - clearOldValue = true, if the old value of query will be cleared before combining in the
    // visibility resolve pass. This flag is only allowed to be false for the first allocation of
    // the render pass or the query that already has an allocated offset.
    // Note: a query might have more than one allocated offset. They will be combined in the final
    // step.
    angle::Result allocateQueryOffset(ContextMtl *contextMtl, QueryMtl *query, bool clearOldValue);
    // Deallocate all offsets used for a query.
    void deallocateQueryOffset(ContextMtl *contextMtl, QueryMtl *query);
    // Retrieve a buffer that will contain the visibility results of all allocated queries for
    // a render pass
    const BufferRef &getRenderPassVisibilityPoolBuffer() const { return mRenderPassResultsPool; }
    size_t getNumRenderPassAllocatedQueries() const { return mAllocatedQueries.size(); }
    // This function is called at the end of render pass
    void resolveVisibilityResults(ContextMtl *contextMtl);
    // Clear visibility pool buffer to drop previous results
    void prepareRenderPassVisibilityPoolBuffer(ContextMtl *contextMtl);

  private:
    // Buffer to hold the visibility results for current render pass
    BufferRef mRenderPassResultsPool;

    // List of allocated queries per render pass
    std::vector<QueryMtl *> mAllocatedQueries;

    bool mResetFirstQuery = false;
    bool mUsed            = false;
};

}  // namespace mtl
}  // namespace rx

#endif /* LIBANGLE_RENDERER_METAL_MTL_OCCLUSION_QUERY_POOL_H_ */
