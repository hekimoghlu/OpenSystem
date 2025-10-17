/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 13, 2025.
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
#include "InspectorShaderProgram.h"

#if ENABLE(WEBGL)

#include "InspectorCanvas.h"
#include "JSExecState.h"
#include "ScriptExecutionContext.h"
#include "WebGLProgram.h"
#include "WebGLRenderingContextBase.h"
#include "WebGLSampler.h"
#include "WebGLShader.h"
#include <JavaScriptCore/ConsoleMessage.h>
#include <JavaScriptCore/IdentifiersFactory.h>
#include <JavaScriptCore/ScriptCallStack.h>
#include <JavaScriptCore/ScriptCallStackFactory.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

using namespace Inspector;

Ref<InspectorShaderProgram> InspectorShaderProgram::create(WebGLProgram& program, InspectorCanvas& inspectorCanvas)
{
    return adoptRef(*new InspectorShaderProgram(program, inspectorCanvas));
}

InspectorShaderProgram::InspectorShaderProgram(WebGLProgram& program, InspectorCanvas& inspectorCanvas)
    : m_identifier(makeString("program:"_s, IdentifiersFactory::createIdentifier()))
    , m_canvas(inspectorCanvas)
    , m_program(program)
{
    ASSERT(is<WebGLRenderingContextBase>(m_canvas.canvasContext()));
}

static WebGLShader* shaderForType(WebGLProgram& program, Inspector::Protocol::Canvas::ShaderType shaderType)
{
    switch (shaderType) {
    case Inspector::Protocol::Canvas::ShaderType::Fragment:
        return program.getAttachedShader(GraphicsContextGL::FRAGMENT_SHADER);

    case Inspector::Protocol::Canvas::ShaderType::Vertex:
        return program.getAttachedShader(GraphicsContextGL::VERTEX_SHADER);

    // Compute shaders are a WebGPU concept.
    case Inspector::Protocol::Canvas::ShaderType::Compute:
        return nullptr;
    }

    ASSERT_NOT_REACHED();
    return nullptr;
}

String InspectorShaderProgram::requestShaderSource(Inspector::Protocol::Canvas::ShaderType shaderType)
{
    auto* shader = shaderForType(m_program, shaderType);
    if (!shader)
        return String();
    return shader->getSource();
}

bool InspectorShaderProgram::updateShader(Inspector::Protocol::Canvas::ShaderType shaderType, const String& source)
{
    auto* shader = shaderForType(m_program, shaderType);
    if (!shader)
        return false;
    auto* context = dynamicDowncast<WebGLRenderingContextBase>(m_canvas.canvasContext());
    if (!context)
        return false;
    context->shaderSource(*shader, source);
    context->compileShader(*shader);
    auto compileStatus = context->getShaderParameter(*shader, GraphicsContextGL::COMPILE_STATUS);
    if (!std::holds_alternative<bool>(compileStatus))
        return false;
    if (std::get<bool>(compileStatus))
        context->linkProgramWithoutInvalidatingAttribLocations(m_program);
    else {
        auto errors = context->getShaderInfoLog(*shader);
        auto* scriptContext = m_canvas.scriptExecutionContext();
        for (auto error : StringView(errors).split('\n')) {
            auto message = makeString("WebGL: "_s, error);
            scriptContext->addConsoleMessage(makeUnique<ConsoleMessage>(MessageSource::Rendering, MessageType::Log, MessageLevel::Error, message));
        }
    }
    return true;
}

Ref<Inspector::Protocol::Canvas::ShaderProgram> InspectorShaderProgram::buildObjectForShaderProgram()
{
    return Inspector::Protocol::Canvas::ShaderProgram::create()
        .setProgramId(m_identifier)
        .setProgramType(Inspector::Protocol::Canvas::ProgramType::Render)
        .setCanvasId(m_canvas.identifier())
        .release();
}

} // namespace WebCore

#endif // ENABLE(WEBGL)
