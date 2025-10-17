/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 9, 2022.
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

// VertexDeclarationCache.h: Defines a helper class to construct and cache vertex declarations.

#ifndef LIBANGLE_RENDERER_D3D_D3D9_VERTEXDECLARATIONCACHE_H_
#define LIBANGLE_RENDERER_D3D_D3D9_VERTEXDECLARATIONCACHE_H_

#include "libANGLE/Error.h"
#include "libANGLE/renderer/d3d/VertexDataManager.h"

namespace gl
{
class VertexDataManager;
class ProgramExecutable;
}  // namespace gl

namespace rx
{
class VertexDeclarationCache
{
  public:
    VertexDeclarationCache();
    ~VertexDeclarationCache();

    angle::Result applyDeclaration(const gl::Context *context,
                                   IDirect3DDevice9 *device,
                                   const std::vector<TranslatedAttribute> &attributes,
                                   gl::ProgramExecutable *executable,
                                   GLint start,
                                   GLsizei instances,
                                   GLsizei *repeatDraw);

    void markStateDirty();

  private:
    UINT mMaxLru;

    enum
    {
        NUM_VERTEX_DECL_CACHE_ENTRIES = 32
    };

    struct VBData
    {
        unsigned int serial;
        unsigned int stride;
        unsigned int offset;
    };

    VBData mAppliedVBs[gl::MAX_VERTEX_ATTRIBS];
    IDirect3DVertexDeclaration9 *mLastSetVDecl;
    bool mInstancingEnabled;

    struct VertexDeclCacheEntry
    {
        D3DVERTEXELEMENT9 cachedElements[gl::MAX_VERTEX_ATTRIBS + 1];
        UINT lruCount;
        IDirect3DVertexDeclaration9 *vertexDeclaration;
    } mVertexDeclCache[NUM_VERTEX_DECL_CACHE_ENTRIES];
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_D3D_D3D9_VERTEXDECLARATIONCACHE_H_
