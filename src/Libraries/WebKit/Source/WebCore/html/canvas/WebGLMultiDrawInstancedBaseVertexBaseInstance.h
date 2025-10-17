/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 7, 2023.
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
#pragma once

#include "WebGLExtension.h"
#include "WebGLRenderingContextBase.h"
#include <JavaScriptCore/TypedArrays.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class WebGLMultiDrawInstancedBaseVertexBaseInstance final : public WebGLExtension<WebGLRenderingContextBase> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WebGLMultiDrawInstancedBaseVertexBaseInstance);
    WTF_MAKE_NONCOPYABLE(WebGLMultiDrawInstancedBaseVertexBaseInstance);
public:
    using Int32List = WebGLRenderingContextBase::TypedList<Int32Array, int32_t>;
    using Uint32List = WebGLRenderingContextBase::TypedList<Uint32Array, uint32_t>;

    explicit WebGLMultiDrawInstancedBaseVertexBaseInstance(WebGLRenderingContextBase&);
    ~WebGLMultiDrawInstancedBaseVertexBaseInstance();

    static bool supported(GraphicsContextGL&);

    void multiDrawArraysInstancedBaseInstanceWEBGL(GCGLenum mode, Int32List&& firstsList, GCGLuint firstsOffset, Int32List&& countsList, GCGLuint countsOffset, Int32List&& instanceCountsList, GCGLuint instanceCountsOffset, Uint32List&& baseInstancesList, GCGLuint baseInstancesOffset, GCGLsizei drawcount);

    void multiDrawElementsInstancedBaseVertexBaseInstanceWEBGL(GCGLenum mode, Int32List&& countsList, GCGLuint countsOffset, GCGLenum type, Int32List&& offsetsList, GCGLuint offsetsOffset, Int32List&& instanceCountsList, GCGLuint instanceCountsOffset, Int32List&& baseVerticesList, GCGLuint baseVerticesOffset, Uint32List&& baseInstancesList, GCGLuint baseInstancesOffset, GCGLsizei drawcount);

private:
    bool validateDrawcount(WebGLRenderingContextBase&, ASCIILiteral functionName, GCGLsizei drawcount);
    bool validateOffset(WebGLRenderingContextBase&, ASCIILiteral functionName, ASCIILiteral outOfBoundsDescription, GCGLsizei, GCGLuint offset, GCGLsizei drawcount);
};

} // namespace WebCore
