/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 6, 2023.
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
#include "InspectorCanvasCallTracer.h"

#include "CSSStyleImageValue.h"
#include "CanvasBase.h"
#include "CanvasGradient.h"
#include "CanvasPattern.h"
#include "CanvasRenderingContext.h"
#include "CanvasRenderingContext2D.h"
#include "DOMMatrix2DInit.h"
#include "DOMPointInit.h"
#include "Element.h"
#include "HTMLCanvasElement.h"
#include "HTMLImageElement.h"
#include "HTMLVideoElement.h"
#include "ImageData.h"
#include "ImageDataSettings.h"
#include "InspectorCanvasAgent.h"
#include "InspectorInstrumentation.h"
#include "InstrumentingAgents.h"
#include "OffscreenCanvas.h"
#include "Path2D.h"
#include "RecordingSwizzleType.h"
#include "WebGL2RenderingContext.h"
#include "WebGLBuffer.h"
#include "WebGLFramebuffer.h"
#include "WebGLProgram.h"
#include "WebGLQuery.h"
#include "WebGLRenderbuffer.h"
#include "WebGLRenderingContextBase.h"
#include "WebGLSampler.h"
#include "WebGLShader.h"
#include "WebGLSync.h"
#include "WebGLTexture.h"
#include "WebGLTransformFeedback.h"
#include "WebGLUniformLocation.h"
#include "WebGLVertexArrayObject.h"
#include <JavaScriptCore/ArrayBuffer.h>
#include <JavaScriptCore/ArrayBufferView.h>
#include <JavaScriptCore/TypedArrays.h>
#include <variant>
#include <wtf/RefPtr.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

static InspectorCanvasAgent* enabledCanvasAgent(CanvasRenderingContext& canvasRenderingContext)
{
    ASSERT(InspectorInstrumentationPublic::hasFrontends());

    auto* agents = InspectorInstrumentation::instrumentingAgents(canvasRenderingContext.canvasBase().scriptExecutionContext());
    ASSERT(agents);
    if (!agents)
        return nullptr;

    ASSERT(agents->enabledCanvasAgent());
    return agents->enabledCanvasAgent();
}

#define PROCESS_ARGUMENT_DEFINITION(ArgumentType) \
std::optional<InspectorCanvasCallTracer::ProcessedArgument> InspectorCanvasCallTracer::processArgument(CanvasRenderingContext& canvasRenderingContext, ArgumentType argument) \
{ \
    if (auto* canvasAgent = enabledCanvasAgent(canvasRenderingContext)) \
        return canvasAgent->processArgument(canvasRenderingContext, argument); \
    return std::nullopt; \
} \
// end of PROCESS_ARGUMENT_DEFINITION
    FOR_EACH_INSPECTOR_CANVAS_CALL_TRACER_ARGUMENT(PROCESS_ARGUMENT_DEFINITION)
#undef PROCESS_ARGUMENT_DEFINITION

void InspectorCanvasCallTracer::recordAction(CanvasRenderingContext& canvasRenderingContext, String&& name, InspectorCanvasCallTracer::ProcessedArguments&& arguments)
{
    if (auto* canvasAgent = enabledCanvasAgent(canvasRenderingContext))
        canvasAgent->recordAction(canvasRenderingContext, WTFMove(name), WTFMove(arguments));
}

std::optional<InspectorCanvasCallTracer::ProcessedArgument> InspectorCanvasCallTracer::processArgument(const CanvasBase& canvasBase, uint32_t argument)
{
    ASSERT(canvasBase.renderingContext());
    return processArgument(*canvasBase.renderingContext(), argument);
}

void InspectorCanvasCallTracer::recordAction(const CanvasBase& canvasBase, String&& name, InspectorCanvasCallTracer::ProcessedArguments&& arguments)
{
    ASSERT(canvasBase.renderingContext());
    recordAction(*canvasBase.renderingContext(), WTFMove(name), WTFMove(arguments));
}

} // namespace WebCore
