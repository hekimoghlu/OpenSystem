/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 5, 2021.
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
#include "WebGLUtilities.h"

#include "InspectorInstrumentation.h"

namespace WebCore {

bool ScopedInspectorShaderProgramHighlight::shouldApply(WebGLRenderingContextBase& context)
{
    if (LIKELY(!context.m_currentProgram || !InspectorInstrumentation::isWebGLProgramHighlighted(context, *context.m_currentProgram)))
        return false;
    if (context.m_framebufferBinding)
        return false;
    return true;
}

void ScopedInspectorShaderProgramHighlight::showHighlight()
{
    Ref gl = *m_context->graphicsContextGL();
    // When OES_draw_buffers_indexed extension is enabled,
    // these state queries return the state for draw buffer 0.
    // Constant blend color is always the same for all draw buffers.
    gl->getFloatv(GraphicsContextGL::BLEND_COLOR, m_savedBlend.color);
    m_savedBlend.equationRGB = gl->getInteger(GraphicsContextGL::BLEND_EQUATION_RGB);
    m_savedBlend.equationAlpha = gl->getInteger(GraphicsContextGL::BLEND_EQUATION_ALPHA);
    m_savedBlend.srcRGB = gl->getInteger(GraphicsContextGL::BLEND_SRC_RGB);
    m_savedBlend.dstRGB = gl->getInteger(GraphicsContextGL::BLEND_DST_RGB);
    m_savedBlend.srcAlpha = gl->getInteger(GraphicsContextGL::BLEND_SRC_ALPHA);
    m_savedBlend.dstAlpha = gl->getInteger(GraphicsContextGL::BLEND_DST_ALPHA);
    m_savedBlend.enabled = gl->isEnabled(GraphicsContextGL::BLEND);

    static const GCGLfloat red = 111.0 / 255.0;
    static const GCGLfloat green = 168.0 / 255.0;
    static const GCGLfloat blue = 220.0 / 255.0;
    static const GCGLfloat alpha = 2.0 / 3.0;
    gl->blendColor(red, green, blue, alpha);

    if (m_context->m_oesDrawBuffersIndexed) {
        gl->enableiOES(GraphicsContextGL::BLEND, 0);
        gl->blendEquationiOES(0, GraphicsContextGL::FUNC_ADD);
        gl->blendFunciOES(0, GraphicsContextGL::CONSTANT_COLOR, GraphicsContextGL::ONE_MINUS_SRC_ALPHA);
    } else {
        gl->enable(GraphicsContextGL::BLEND);
        gl->blendEquation(GraphicsContextGL::FUNC_ADD);
        gl->blendFunc(GraphicsContextGL::CONSTANT_COLOR, GraphicsContextGL::ONE_MINUS_SRC_ALPHA);
    }
}

void ScopedInspectorShaderProgramHighlight::hideHighlight()
{
    Ref gl = *m_context->graphicsContextGL();
    gl->blendColor(m_savedBlend.color[0], m_savedBlend.color[1], m_savedBlend.color[2], m_savedBlend.color[3]);

    if (m_context->m_oesDrawBuffersIndexed) {
        gl->blendEquationSeparateiOES(0, m_savedBlend.equationRGB, m_savedBlend.equationAlpha);
        gl->blendFuncSeparateiOES(0, m_savedBlend.srcRGB, m_savedBlend.dstRGB, m_savedBlend.srcAlpha, m_savedBlend.dstAlpha);
        if (!m_savedBlend.enabled)
            gl->disableiOES(GraphicsContextGL::BLEND, 0);
    } else {
        gl->blendEquationSeparate(m_savedBlend.equationRGB, m_savedBlend.equationAlpha);
        gl->blendFuncSeparate(m_savedBlend.srcRGB, m_savedBlend.dstRGB, m_savedBlend.srcAlpha, m_savedBlend.dstAlpha);
        if (!m_savedBlend.enabled)
            gl->disable(GraphicsContextGL::BLEND);
    }
}

}

#endif
