/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 15, 2024.
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

class WebGLMultiDraw final : public WebGLExtension<WebGLRenderingContextBase> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WebGLMultiDraw);
    WTF_MAKE_NONCOPYABLE(WebGLMultiDraw);
public:
    using Int32List = WebGLRenderingContextBase::TypedList<Int32Array, int32_t>;

    explicit WebGLMultiDraw(WebGLRenderingContextBase&);
    ~WebGLMultiDraw();

    static bool supported(GraphicsContextGL&);

    void multiDrawArraysWEBGL(GCGLenum mode, Int32List&& firstsList, GCGLuint firstsOffset, Int32List&& countsList, GCGLuint countsOffset, GCGLsizei drawcount);

    void multiDrawElementsWEBGL(GCGLenum mode, Int32List&& countsList, GCGLuint countsOffset, GCGLenum type, Int32List&& offsetsList, GCGLuint offsetsOffset, GCGLsizei drawcount);

    void multiDrawArraysInstancedWEBGL(GCGLenum mode, Int32List&& firstsList, GCGLuint firstsOffset, Int32List&& countsList, GCGLuint countsOffset, Int32List&& instanceCountsList, GCGLuint instanceCountsOffset, GCGLsizei drawcount);

    void multiDrawElementsInstancedWEBGL(GCGLenum mode, Int32List&& countsList, GCGLuint countsOffset, GCGLenum type, Int32List&& offsetsList, GCGLuint offsetsOffset, Int32List&& instanceCountsList, GCGLuint instanceCountsOffset, GCGLsizei drawcount);

private:
    bool validateDrawcount(WebGLRenderingContextBase&, ASCIILiteral functionName, GCGLsizei drawcount);
    bool validateOffset(WebGLRenderingContextBase&, ASCIILiteral functionName, ASCIILiteral outOfBoundsDescription, GCGLsizei, GCGLuint offset, GCGLsizei drawcount);
};

} // namespace WebCore
