/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 23, 2023.
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
#include "WebGLDrawInstancedBaseVertexBaseInstance.h"

#include "InspectorInstrumentation.h"
#include "WebGLUtilities.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WebGLDrawInstancedBaseVertexBaseInstance);

WebGLDrawInstancedBaseVertexBaseInstance::WebGLDrawInstancedBaseVertexBaseInstance(WebGLRenderingContextBase& context)
    : WebGLExtension(context, WebGLExtensionName::WebGLDrawInstancedBaseVertexBaseInstance)
{
    context.protectedGraphicsContextGL()->ensureExtensionEnabled("GL_ANGLE_base_vertex_base_instance"_s);
}

WebGLDrawInstancedBaseVertexBaseInstance::~WebGLDrawInstancedBaseVertexBaseInstance() = default;

bool WebGLDrawInstancedBaseVertexBaseInstance::supported(GraphicsContextGL& context)
{
    return context.supportsExtension("GL_ANGLE_base_vertex_base_instance"_s);
}

void WebGLDrawInstancedBaseVertexBaseInstance::drawArraysInstancedBaseInstanceWEBGL(GCGLenum mode, GCGLint first, GCGLsizei count, GCGLsizei instanceCount, GCGLuint baseInstance)
{
    if (isContextLost())
        return;
    auto& context = this->context();

    if (!context.validateVertexArrayObject("drawArraysInstancedBaseInstanceWEBGL"_s))
        return;

    if (context.m_currentProgram && InspectorInstrumentation::isWebGLProgramDisabled(context, *context.m_currentProgram))
        return;

    context.clearIfComposited(WebGLRenderingContextBase::CallerTypeDrawOrClear);

    {
        ScopedInspectorShaderProgramHighlight scopedHighlight { context };

        context.protectedGraphicsContextGL()->drawArraysInstancedBaseInstanceANGLE(mode, first, count, instanceCount, baseInstance);
    }

    context.markContextChangedAndNotifyCanvasObserver();
}

void WebGLDrawInstancedBaseVertexBaseInstance::drawElementsInstancedBaseVertexBaseInstanceWEBGL(GCGLenum mode, GCGLsizei count, GCGLenum type, GCGLintptr offset, GCGLsizei instanceCount, GCGLint baseVertex, GCGLuint baseInstance)
{
    if (isContextLost())
        return;
    auto& context = this->context();

    if (!context.validateVertexArrayObject("drawElementsInstancedBaseVertexBaseInstanceWEBGL"_s))
        return;

    if (context.m_currentProgram && InspectorInstrumentation::isWebGLProgramDisabled(context, *context.m_currentProgram))
        return;

    context.clearIfComposited(WebGLRenderingContextBase::CallerTypeDrawOrClear);

    {
        ScopedInspectorShaderProgramHighlight scopedHighlight { context };

        context.protectedGraphicsContextGL()->drawElementsInstancedBaseVertexBaseInstanceANGLE(mode, count, type, offset, instanceCount, baseVertex, baseInstance);
    }

    context.markContextChangedAndNotifyCanvasObserver();
}

} // namespace WebCore

#endif // ENABLE(WEBGL)
