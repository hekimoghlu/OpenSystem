/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 13, 2021.
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
#include "WebGLMultiDrawInstancedBaseVertexBaseInstance.h"

#include "InspectorInstrumentation.h"
#include "WebGLUtilities.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WebGLMultiDrawInstancedBaseVertexBaseInstance);

WebGLMultiDrawInstancedBaseVertexBaseInstance::WebGLMultiDrawInstancedBaseVertexBaseInstance(WebGLRenderingContextBase& context)
    : WebGLExtension(context, WebGLExtensionName::WebGLMultiDrawInstancedBaseVertexBaseInstance)
{
    context.protectedGraphicsContextGL()->ensureExtensionEnabled("GL_ANGLE_base_vertex_base_instance"_s);

    // Spec requires WEBGL_multi_draw to be turned on implicitly here.
    // Enable it both in the backend and in WebKit.
    context.getExtension("WEBGL_multi_draw"_s);
}

WebGLMultiDrawInstancedBaseVertexBaseInstance::~WebGLMultiDrawInstancedBaseVertexBaseInstance() = default;

bool WebGLMultiDrawInstancedBaseVertexBaseInstance::supported(GraphicsContextGL& context)
{
    return context.supportsExtension("GL_ANGLE_base_vertex_base_instance"_s)
        && context.supportsExtension("GL_ANGLE_multi_draw"_s);
}

void WebGLMultiDrawInstancedBaseVertexBaseInstance::multiDrawArraysInstancedBaseInstanceWEBGL(GCGLenum mode, Int32List&& firstsList, GCGLuint firstsOffset, Int32List&& countsList, GCGLuint countsOffset, Int32List&& instanceCountsList, GCGLuint instanceCountsOffset, Uint32List&& baseInstancesList, GCGLuint baseInstancesOffset, GCGLsizei drawcount)
{
    if (isContextLost())
        return;
    auto& context = this->context();

    if (!validateDrawcount(context, "multiDrawArraysInstancedBaseInstanceWEBGL"_s, drawcount)
        || !validateOffset(context, "multiDrawArraysInstancedBaseInstanceWEBGL"_s, "firstsOffset out of bounds"_s, firstsList.length(), firstsOffset, drawcount)
        || !validateOffset(context, "multiDrawArraysInstancedBaseInstanceWEBGL"_s, "countsOffset out of bounds"_s, countsList.length(), countsOffset, drawcount)
        || !validateOffset(context, "multiDrawArraysInstancedBaseInstanceWEBGL"_s, "instanceCountsOffset out of bounds"_s, instanceCountsList.length(), instanceCountsOffset, drawcount)
        || !validateOffset(context, "multiDrawArraysInstancedBaseInstanceWEBGL"_s, "baseInstancesOffset out of bounds"_s, baseInstancesList.length(), baseInstancesOffset, drawcount)) {
        return;
    }

    if (!context.validateVertexArrayObject("multiDrawArraysInstancedBaseInstanceWEBGL"_s))
        return;

    if (context.m_currentProgram && InspectorInstrumentation::isWebGLProgramDisabled(context, *context.m_currentProgram))
        return;

    context.clearIfComposited(WebGLRenderingContextBase::CallerTypeDrawOrClear);

    {
        ScopedInspectorShaderProgramHighlight scopedHighlight { context };

        context.protectedGraphicsContextGL()->multiDrawArraysInstancedBaseInstanceANGLE(mode, GCGLSpanTuple { firstsList.span().subspan(firstsOffset).data(), countsList.span().subspan(countsOffset).data(), instanceCountsList.span().subspan(instanceCountsOffset).data(), baseInstancesList.span().subspan(baseInstancesOffset).data(), static_cast<size_t>(drawcount) });
    }

    context.markContextChangedAndNotifyCanvasObserver();
}

void WebGLMultiDrawInstancedBaseVertexBaseInstance::multiDrawElementsInstancedBaseVertexBaseInstanceWEBGL(GCGLenum mode, Int32List&& countsList, GCGLuint countsOffset, GCGLenum type, Int32List&& offsetsList, GCGLuint offsetsOffset, Int32List&& instanceCountsList, GCGLuint instanceCountsOffset, Int32List&& baseVerticesList, GCGLuint baseVerticesOffset, Uint32List&& baseInstancesList, GCGLuint baseInstancesOffset, GCGLsizei drawcount)
{
    if (isContextLost())
        return;
    auto& context = this->context();

    if (!validateDrawcount(context, "multiDrawElementsInstancedBaseVertexBaseInstanceWEBGL"_s, drawcount)
        || !validateOffset(context, "multiDrawElementsInstancedBaseVertexBaseInstanceWEBGL"_s, "countsOffset out of bounds"_s, countsList.length(), countsOffset, drawcount)
        || !validateOffset(context, "multiDrawElementsInstancedBaseVertexBaseInstanceWEBGL"_s, "offsetsOffset out of bounds"_s, offsetsList.length(), offsetsOffset, drawcount)
        || !validateOffset(context, "multiDrawElementsInstancedBaseVertexBaseInstanceWEBGL"_s, "instanceCountsOffset out of bounds"_s, instanceCountsList.length(), instanceCountsOffset, drawcount)
        || !validateOffset(context, "multiDrawElementsInstancedBaseVertexBaseInstanceWEBGL"_s, "baseVerticesOffset out of bounds"_s, baseVerticesList.length(), baseVerticesOffset, drawcount)
        || !validateOffset(context, "multiDrawElementsInstancedBaseVertexBaseInstanceWEBGL"_s, "baseInstancesOffset out of bounds"_s, baseInstancesList.length(), baseInstancesOffset, drawcount)) {
        return;
    }

    if (!context.validateVertexArrayObject("multiDrawElementsInstancedBaseVertexBaseInstanceWEBGL"_s))
        return;

    if (context.m_currentProgram && InspectorInstrumentation::isWebGLProgramDisabled(context, *context.m_currentProgram))
        return;

    context.clearIfComposited(WebGLRenderingContextBase::CallerTypeDrawOrClear);

    {
        ScopedInspectorShaderProgramHighlight scopedHighlight { context };

        context.protectedGraphicsContextGL()->multiDrawElementsInstancedBaseVertexBaseInstanceANGLE(mode, GCGLSpanTuple { countsList.span().subspan(countsOffset).data(), offsetsList.span().subspan(offsetsOffset).data(), instanceCountsList.span().subspan(instanceCountsOffset).data(), baseVerticesList.span().subspan(baseVerticesOffset).data(), baseInstancesList.span().subspan(baseInstancesOffset).data(), static_cast<size_t>(drawcount) }, type);
    }

    context.markContextChangedAndNotifyCanvasObserver();
}

bool WebGLMultiDrawInstancedBaseVertexBaseInstance::validateDrawcount(WebGLRenderingContextBase& context, ASCIILiteral functionName, GCGLsizei drawcount)
{
    if (drawcount < 0) {
        context.synthesizeGLError(GraphicsContextGL::INVALID_VALUE, functionName, "negative drawcount"_s);
        return false;
    }

    return true;
}

bool WebGLMultiDrawInstancedBaseVertexBaseInstance::validateOffset(WebGLRenderingContextBase& context, ASCIILiteral functionName, ASCIILiteral outOfBoundsDescription, GCGLsizei size, GCGLuint offset, GCGLsizei drawcount)
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
