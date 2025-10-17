/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 11, 2023.
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
#include "config.h"

#if ENABLE(WEBGL)
#include "WebGLMultiDraw.h"

#include "InspectorInstrumentation.h"
#include "WebGLUtilities.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WebGLMultiDraw);

WebGLMultiDraw::WebGLMultiDraw(WebGLRenderingContextBase& context)
    : WebGLExtension(context, WebGLExtensionName::WebGLMultiDraw)
{
    context.protectedGraphicsContextGL()->ensureExtensionEnabled("GL_ANGLE_multi_draw"_s);

    // Spec requires ANGLE_instanced_arrays to be turned on implicitly here.
    // Enable it both in the backend and in WebKit.
    if (context.isWebGL1())
        context.getExtension("ANGLE_instanced_arrays"_s);
}

WebGLMultiDraw::~WebGLMultiDraw() = default;

bool WebGLMultiDraw::supported(GraphicsContextGL& context)
{
    return context.supportsExtension("GL_ANGLE_multi_draw"_s)
        && context.supportsExtension("GL_ANGLE_instanced_arrays"_s);
}

void WebGLMultiDraw::multiDrawArraysWEBGL(GCGLenum mode, Int32List&& firstsList, GCGLuint firstsOffset, Int32List&& countsList, GCGLuint countsOffset, GCGLsizei drawcount)
{
    if (isContextLost())
        return;
    auto& context = this->context();

    if (!validateDrawcount(context, "multiDrawArraysWEBGL"_s, drawcount)
        || !validateOffset(context, "multiDrawArraysWEBGL"_s, "firstsOffset out of bounds"_s, firstsList.length(), firstsOffset, drawcount)
        || !validateOffset(context, "multiDrawArraysWEBGL"_s, "countsOffset out of bounds"_s, countsList.length(), countsOffset, drawcount)) {
        return;
    }

    if (!context.validateVertexArrayObject("multiDrawArraysWEBGL"_s))
        return;

    if (context.m_currentProgram && InspectorInstrumentation::isWebGLProgramDisabled(context, *context.m_currentProgram))
        return;

    context.clearIfComposited(WebGLRenderingContextBase::CallerTypeDrawOrClear);

    {
        ScopedInspectorShaderProgramHighlight scopedHighlight { context };

        context.protectedGraphicsContextGL()->multiDrawArraysANGLE(mode, GCGLSpanTuple { firstsList.span().subspan(firstsOffset).data(), countsList.span().subspan(countsOffset).data(), static_cast<size_t>(drawcount) });
    }

    context.markContextChangedAndNotifyCanvasObserver();
}

void WebGLMultiDraw::multiDrawArraysInstancedWEBGL(GCGLenum mode, Int32List&& firstsList, GCGLuint firstsOffset, Int32List&& countsList, GCGLuint countsOffset, Int32List&& instanceCountsList, GCGLuint instanceCountsOffset, GCGLsizei drawcount)
{
    if (isContextLost())
        return;
    auto& context = this->context();

    if (!validateDrawcount(context, "multiDrawArraysInstancedWEBGL"_s, drawcount)
        || !validateOffset(context, "multiDrawArraysInstancedWEBGL"_s, "firstsOffset out of bounds"_s, firstsList.length(), firstsOffset, drawcount)
        || !validateOffset(context, "multiDrawArraysInstancedWEBGL"_s, "countsOffset out of bounds"_s, countsList.length(), countsOffset, drawcount)
        || !validateOffset(context, "multiDrawArraysInstancedWEBGL"_s, "instanceCountsOffset out of bounds"_s, instanceCountsList.length(), instanceCountsOffset, drawcount)) {
        return;
    }

    if (!context.validateVertexArrayObject("multiDrawArraysInstancedWEBGL"_s))
        return;

    if (context.m_currentProgram && InspectorInstrumentation::isWebGLProgramDisabled(context, *context.m_currentProgram))
        return;

    context.clearIfComposited(WebGLRenderingContextBase::CallerTypeDrawOrClear);

    {
        ScopedInspectorShaderProgramHighlight scopedHighlight { context };

        context.protectedGraphicsContextGL()->multiDrawArraysInstancedANGLE(mode, GCGLSpanTuple { firstsList.span().subspan(firstsOffset).data(), countsList.span().subspan(countsOffset).data(), instanceCountsList.span().subspan(instanceCountsOffset).data(), static_cast<size_t>(drawcount) });
    }

    context.markContextChangedAndNotifyCanvasObserver();
}

void WebGLMultiDraw::multiDrawElementsWEBGL(GCGLenum mode, Int32List&& countsList, GCGLuint countsOffset, GCGLenum type, Int32List&& offsetsList, GCGLuint offsetsOffset, GCGLsizei drawcount)
{
    if (isContextLost())
        return;
    auto& context = this->context();

    if (!validateDrawcount(context, "multiDrawElementsWEBGL"_s, drawcount)
        || !validateOffset(context, "multiDrawElementsWEBGL"_s, "countsOffset out of bounds"_s, countsList.length(), countsOffset, drawcount)
        || !validateOffset(context, "multiDrawElementsWEBGL"_s, "offsetsOffset out of bounds"_s, offsetsList.length(), offsetsOffset, drawcount)) {
        return;
    }

    if (!context.validateVertexArrayObject("multiDrawElementsWEBGL"_s))
        return;

    if (context.m_currentProgram && InspectorInstrumentation::isWebGLProgramDisabled(context, *context.m_currentProgram))
        return;

    context.clearIfComposited(WebGLRenderingContextBase::CallerTypeDrawOrClear);

    {
        ScopedInspectorShaderProgramHighlight scopedHighlight { context };

        context.protectedGraphicsContextGL()->multiDrawElementsANGLE(mode, GCGLSpanTuple { countsList.span().subspan(countsOffset).data(), offsetsList.span().subspan(offsetsOffset).data(), static_cast<size_t>(drawcount) }, type);
    }

    context.markContextChangedAndNotifyCanvasObserver();
}

void WebGLMultiDraw::multiDrawElementsInstancedWEBGL(GCGLenum mode, Int32List&& countsList, GCGLuint countsOffset, GCGLenum type, Int32List&& offsetsList, GCGLuint offsetsOffset, Int32List&& instanceCountsList, GCGLuint instanceCountsOffset, GCGLsizei drawcount)
{
    if (isContextLost())
        return;
    auto& context = this->context();

    if (!validateDrawcount(context, "multiDrawElementsInstancedWEBGL"_s, drawcount)
        || !validateOffset(context, "multiDrawElementsInstancedWEBGL"_s, "countsOffset out of bounds"_s, countsList.length(), countsOffset, drawcount)
        || !validateOffset(context, "multiDrawElementsInstancedWEBGL"_s, "offsetsOffset out of bounds"_s, offsetsList.length(), offsetsOffset, drawcount)
        || !validateOffset(context, "multiDrawElementsInstancedWEBGL"_s, "instanceCountsOffset out of bounds"_s, instanceCountsList.length(), instanceCountsOffset, drawcount)) {
        return;
    }

    if (!context.validateVertexArrayObject("multiDrawElementsInstancedWEBGL"_s))
        return;

    if (context.m_currentProgram && InspectorInstrumentation::isWebGLProgramDisabled(context, *context.m_currentProgram))
        return;

    context.clearIfComposited(WebGLRenderingContextBase::CallerTypeDrawOrClear);

    {
        ScopedInspectorShaderProgramHighlight scopedHighlight { context };

        context.protectedGraphicsContextGL()->multiDrawElementsInstancedANGLE(mode, GCGLSpanTuple { countsList.span().subspan(countsOffset).data(), offsetsList.span().subspan(offsetsOffset).data(), instanceCountsList.span().subspan(instanceCountsOffset).data(), static_cast<size_t>(drawcount) }, type);
    }

    context.markContextChangedAndNotifyCanvasObserver();
}

bool WebGLMultiDraw::validateDrawcount(WebGLRenderingContextBase& context, ASCIILiteral functionName, GCGLsizei drawcount)
{
    if (drawcount < 0) {
        context.synthesizeGLError(GraphicsContextGL::INVALID_VALUE, functionName, "negative drawcount"_s);
        return false;
    }

    return true;
}

bool WebGLMultiDraw::validateOffset(WebGLRenderingContextBase& context, ASCIILiteral functionName, ASCIILiteral outOfBoundsDescription, GCGLsizei size, GCGLuint offset, GCGLsizei drawcount)
{
    if (drawcount > size) {
        context.synthesizeGLError(GraphicsContextGL::INVALID_OPERATION, functionName, "drawcount out of bounds"_s);
        return false;
    }

    if (offset > static_cast<GCGLuint>(size - drawcount)) {
        context.synthesizeGLError(GraphicsContextGL::INVALID_OPERATION, functionName, outOfBoundsDescription);
        return false;
    }

    return true;
}

} // namespace WebCore

#endif // ENABLE(WEBGL)
