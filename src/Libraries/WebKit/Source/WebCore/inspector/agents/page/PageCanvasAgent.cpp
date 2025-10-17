/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 16, 2022.
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
#include "PageCanvasAgent.h"

#include "CSSStyleImageValue.h"
#include "CanvasBase.h"
#include "CanvasGradient.h"
#include "CanvasPattern.h"
#include "CanvasRenderingContext.h"
#include "Document.h"
#include "Element.h"
#include "FrameDestructionObserverInlines.h"
#include "HTMLCanvasElement.h"
#include "HTMLImageElement.h"
#include "HTMLVideoElement.h"
#include "ImageBitmap.h"
#include "ImageData.h"
#include "InspectorCanvas.h"
#include "InspectorDOMAgent.h"
#include "InstrumentingAgents.h"
#include "LocalFrame.h"
#include "Page.h"
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(OFFSCREEN_CANVAS)
#include "OffscreenCanvas.h"
#endif

namespace WebCore {

using namespace Inspector;

WTF_MAKE_TZONE_ALLOCATED_IMPL(PageCanvasAgent);

PageCanvasAgent::PageCanvasAgent(PageAgentContext& context)
    : InspectorCanvasAgent(context)
    , m_inspectedPage(context.inspectedPage)
{
}

PageCanvasAgent::~PageCanvasAgent() = default;

bool PageCanvasAgent::enabled() const
{
    return m_instrumentingAgents.enabledPageCanvasAgent() == this && InspectorCanvasAgent::enabled();
}

void PageCanvasAgent::internalEnable()
{
    m_instrumentingAgents.setEnabledPageCanvasAgent(this);

    InspectorCanvasAgent::internalEnable();
}

void PageCanvasAgent::internalDisable()
{
    m_instrumentingAgents.setEnabledPageCanvasAgent(nullptr);

    InspectorCanvasAgent::internalDisable();
}

Inspector::Protocol::ErrorStringOr<Inspector::Protocol::DOM::NodeId> PageCanvasAgent::requestNode(const Inspector::Protocol::Canvas::CanvasId& canvasId)
{
    Inspector::Protocol::ErrorString errorString;

    auto inspectorCanvas = assertInspectorCanvas(errorString, canvasId);
    if (!inspectorCanvas)
        return makeUnexpected(errorString);

    auto* node = inspectorCanvas->canvasElement();
    if (!node)
        return makeUnexpected("Missing element of canvas for given canvasId"_s);

    // FIXME: <https://webkit.org/b/213499> Web Inspector: allow DOM nodes to be instrumented at any point, regardless of whether the main document has also been instrumented
    int documentNodeId = m_instrumentingAgents.persistentDOMAgent()->boundNodeId(&node->document());
    if (!documentNodeId)
        return makeUnexpected("Document must have been requested"_s);

    return m_instrumentingAgents.persistentDOMAgent()->pushNodeToFrontend(errorString, documentNodeId, node);
}

Inspector::Protocol::ErrorStringOr<Ref<JSON::ArrayOf<Inspector::Protocol::DOM::NodeId>>> PageCanvasAgent::requestClientNodes(const Inspector::Protocol::Canvas::CanvasId& canvasId)
{
    Inspector::Protocol::ErrorString errorString;

    auto* domAgent = m_instrumentingAgents.persistentDOMAgent();
    if (!domAgent)
        return makeUnexpected("DOM domain must be enabled"_s);

    auto inspectorCanvas = assertInspectorCanvas(errorString, canvasId);
    if (!inspectorCanvas)
        return makeUnexpected(errorString);

    auto clientNodeIds = JSON::ArrayOf<Inspector::Protocol::DOM::NodeId>::create();
    for (auto& clientNode : inspectorCanvas->clientNodes()) {
        // FIXME: <https://webkit.org/b/213499> Web Inspector: allow DOM nodes to be instrumented at any point, regardless of whether the main document has also been instrumented
        if (auto documentNodeId = domAgent->boundNodeId(&clientNode->document()))
            clientNodeIds->addItem(domAgent->pushNodeToFrontend(errorString, documentNodeId, clientNode));
    }
    return clientNodeIds;
}

void PageCanvasAgent::frameNavigated(LocalFrame& frame)
{
    if (frame.isMainFrame()) {
        reset();
        return;
    }

    Vector<InspectorCanvas*> inspectorCanvases;
    for (auto& inspectorCanvas : m_identifierToInspectorCanvas.values()) {
        if (auto* canvasElement = inspectorCanvas->canvasElement()) {
            if (canvasElement->document().frame() == &frame)
                inspectorCanvases.append(inspectorCanvas.get());
        }
    }
    for (auto* inspectorCanvas : inspectorCanvases)
        unbindCanvas(*inspectorCanvas);
}

void PageCanvasAgent::didChangeCSSCanvasClientNodes(CanvasBase& canvasBase)
{
    auto* context = canvasBase.renderingContext();
    if (!context) {
        ASSERT_NOT_REACHED();
        return;
    }

    auto inspectorCanvas = findInspectorCanvas(*context);
    ASSERT(inspectorCanvas);
    if (!inspectorCanvas)
        return;

    m_frontendDispatcher->clientNodesChanged(inspectorCanvas->identifier());
}

bool PageCanvasAgent::matchesCurrentContext(ScriptExecutionContext* scriptExecutionContext) const
{
    auto* document = dynamicDowncast<Document>(scriptExecutionContext);
    if (!document)
        return false;

    // FIXME: <https://webkit.org/b/168475> Web Inspector: Correctly display iframe's WebSockets
    return document->page() == m_inspectedPage.ptr();
}

} // namespace WebCore
