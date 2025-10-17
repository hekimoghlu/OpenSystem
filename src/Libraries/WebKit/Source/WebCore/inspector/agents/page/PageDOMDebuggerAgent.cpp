/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 27, 2025.
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
#include "PageDOMDebuggerAgent.h"

#include "Element.h"
#include "InspectorDOMAgent.h"
#include "InstrumentingAgents.h"
#include "LocalFrame.h"
#include "Node.h"

namespace WebCore {

using namespace Inspector;

PageDOMDebuggerAgent::PageDOMDebuggerAgent(PageAgentContext& context, InspectorDebuggerAgent* debuggerAgent)
    : InspectorDOMDebuggerAgent(context, debuggerAgent)
{
}

PageDOMDebuggerAgent::~PageDOMDebuggerAgent() = default;

bool PageDOMDebuggerAgent::enabled() const
{
    return m_instrumentingAgents.enabledPageDOMDebuggerAgent() == this && InspectorDOMDebuggerAgent::enabled();
}

void PageDOMDebuggerAgent::enable()
{
    m_instrumentingAgents.setEnabledPageDOMDebuggerAgent(this);

    InspectorDOMDebuggerAgent::enable();
}

void PageDOMDebuggerAgent::disable()
{
    m_instrumentingAgents.setEnabledPageDOMDebuggerAgent(nullptr);

    m_domSubtreeModifiedBreakpoints.clear();
    m_domAttributeModifiedBreakpoints.clear();
    m_domNodeRemovedBreakpoints.clear();

    m_pauseOnAllAnimationFramesBreakpoint = nullptr;

    InspectorDOMDebuggerAgent::disable();
}

Inspector::Protocol::ErrorStringOr<void> PageDOMDebuggerAgent::setDOMBreakpoint(Inspector::Protocol::DOM::NodeId nodeId, Inspector::Protocol::DOMDebugger::DOMBreakpointType type, RefPtr<JSON::Object>&& options)
{
    Inspector::Protocol::ErrorString errorString;

    auto* domAgent = m_instrumentingAgents.persistentDOMAgent();
    if (!domAgent)
        return makeUnexpected("DOM domain must be enabled"_s);

    Node* node = domAgent->assertNode(errorString, nodeId);
    if (!node)
        return makeUnexpected(errorString);

    auto breakpoint = InspectorDebuggerAgent::debuggerBreakpointFromPayload(errorString, WTFMove(options));
    if (!breakpoint)
        return makeUnexpected(errorString);

    switch (type) {
    case Inspector::Protocol::DOMDebugger::DOMBreakpointType::SubtreeModified:
        if (!m_domSubtreeModifiedBreakpoints.add(node, breakpoint.releaseNonNull()))
            return makeUnexpected("Breakpoint for given node and given type already exists"_s);
        return { };

    case Inspector::Protocol::DOMDebugger::DOMBreakpointType::AttributeModified:
        if (!m_domAttributeModifiedBreakpoints.add(node, breakpoint.releaseNonNull()))
            return makeUnexpected("Breakpoint for given node and given type already exists"_s);
        return { };

    case Inspector::Protocol::DOMDebugger::DOMBreakpointType::NodeRemoved:
        if (!m_domNodeRemovedBreakpoints.add(node, breakpoint.releaseNonNull()))
            return makeUnexpected("Breakpoint for given node and given type already exists"_s);
        return { };
    }

    ASSERT_NOT_REACHED();
    return makeUnexpected("Not supported"_s);
}

Inspector::Protocol::ErrorStringOr<void> PageDOMDebuggerAgent::removeDOMBreakpoint(Inspector::Protocol::DOM::NodeId nodeId, Inspector::Protocol::DOMDebugger::DOMBreakpointType type)
{
    Inspector::Protocol::ErrorString errorString;

    auto* domAgent = m_instrumentingAgents.persistentDOMAgent();
    if (!domAgent)
        return makeUnexpected("DOM domain must be enabled"_s);

    Node* node = domAgent->assertNode(errorString, nodeId);
    if (!node)
        return makeUnexpected(errorString);

    switch (type) {
    case Inspector::Protocol::DOMDebugger::DOMBreakpointType::SubtreeModified:
        if (!m_domSubtreeModifiedBreakpoints.remove(node))
            return makeUnexpected("Breakpoint for given node and given type missing"_s);
        return { };

    case Inspector::Protocol::DOMDebugger::DOMBreakpointType::AttributeModified:
        if (!m_domAttributeModifiedBreakpoints.remove(node))
            return makeUnexpected("Breakpoint for given node and given type missing"_s);
        return { };

    case Inspector::Protocol::DOMDebugger::DOMBreakpointType::NodeRemoved:
        if (!m_domNodeRemovedBreakpoints.remove(node))
            return makeUnexpected("Breakpoint for given node and given type missing"_s);
        return { };
    }

    ASSERT_NOT_REACHED();
    return makeUnexpected("Not supported"_s);
}

void PageDOMDebuggerAgent::mainFrameNavigated()
{
    InspectorDOMDebuggerAgent::mainFrameNavigated();

    if (m_pauseOnAllAnimationFramesBreakpoint)
        m_pauseOnAllAnimationFramesBreakpoint->resetHitCount();
}

void PageDOMDebuggerAgent::frameDocumentUpdated(LocalFrame& frame)
{
    if (!frame.isMainFrame())
        return;

    m_domSubtreeModifiedBreakpoints.clear();
    m_domAttributeModifiedBreakpoints.clear();
    m_domNodeRemovedBreakpoints.clear();
}


static std::optional<size_t> calculateDistance(Node& child, Node& ancestor)
{
    size_t distance = 0;

    auto* current = &child;
    while (current != &ancestor) {
        ++distance;

        current = InspectorDOMAgent::innerParentNode(current);
        if (!current)
            return std::nullopt;
    }

    return distance;
}

void PageDOMDebuggerAgent::willInsertDOMNode(Node& parent)
{
    if (!m_debuggerAgent->breakpointsActive())
        return;

    if (m_domSubtreeModifiedBreakpoints.isEmpty())
        return;

    std::optional<size_t> closestDistance;
    RefPtr<JSC::Breakpoint> closestBreakpoint;
    RefPtr<Node> closestBreakpointOwner;

    for (auto [breakpointOwner, breakpoint] : m_domSubtreeModifiedBreakpoints) {
        auto distance = calculateDistance(parent, Ref { *breakpointOwner });
        if (!distance)
            continue;

        if (!closestDistance || distance < closestDistance) {
            closestDistance = distance;
            closestBreakpoint = breakpoint.copyRef();
            closestBreakpointOwner = breakpointOwner;
        }
    }

    if (!closestBreakpoint)
        return;

    ASSERT(closestBreakpointOwner);

    auto pauseData = buildPauseDataForDOMBreakpoint(Inspector::Protocol::DOMDebugger::DOMBreakpointType::SubtreeModified, *closestBreakpointOwner);
    pauseData->setBoolean("insertion"_s, true);
    // FIXME: <https://webkit.org/b/213499> Web Inspector: allow DOM nodes to be instrumented at any point, regardless of whether the main document has also been instrumented
    // Include the new child node ID so the frontend can show the node that's about to be inserted.
    m_debuggerAgent->breakProgram(Inspector::DebuggerFrontendDispatcher::Reason::DOM, WTFMove(pauseData), WTFMove(closestBreakpoint));
}

void PageDOMDebuggerAgent::willRemoveDOMNode(Node& node)
{
    if (!m_debuggerAgent->breakpointsActive())
        return;

    if (m_domNodeRemovedBreakpoints.isEmpty() && m_domSubtreeModifiedBreakpoints.isEmpty())
        return;

    std::optional<size_t> closestDistance;
    RefPtr<JSC::Breakpoint> closestBreakpoint;
    std::optional<Inspector::Protocol::DOMDebugger::DOMBreakpointType> closestBreakpointType;
    Node* closestBreakpointOwner = nullptr;

    for (auto [breakpointOwner, breakpoint] : m_domNodeRemovedBreakpoints) {
        auto distance = calculateDistance(*breakpointOwner, node);
        if (!distance)
            continue;

        if (!closestDistance || distance < closestDistance) {
            closestDistance = distance;
            closestBreakpoint = breakpoint.copyRef();
            closestBreakpointType = Inspector::Protocol::DOMDebugger::DOMBreakpointType::NodeRemoved;
            closestBreakpointOwner = breakpointOwner;
        }
    }

    if (!closestBreakpoint) {
        for (auto [breakpointOwner, breakpoint] : m_domSubtreeModifiedBreakpoints) {
            auto distance = calculateDistance(node, *breakpointOwner);
            if (!distance)
                continue;

            if (!closestDistance || distance < closestDistance) {
                closestDistance = distance;
                closestBreakpoint = breakpoint.copyRef();
                closestBreakpointType = Inspector::Protocol::DOMDebugger::DOMBreakpointType::SubtreeModified;
                closestBreakpointOwner = breakpointOwner;
            }
        }
    }

    if (!closestBreakpoint)
        return;

    ASSERT(closestBreakpointType);
    ASSERT(closestBreakpointOwner);

    auto pauseData = buildPauseDataForDOMBreakpoint(*closestBreakpointType, *closestBreakpointOwner);
    if (auto* domAgent = m_instrumentingAgents.persistentDOMAgent()) {
        if (&node != closestBreakpointOwner) {
            if (auto targetNodeId = domAgent->pushNodeToFrontend(&node))
                pauseData->setInteger("targetNodeId"_s, targetNodeId);
        }
    }
    m_debuggerAgent->breakProgram(Inspector::DebuggerFrontendDispatcher::Reason::DOM, WTFMove(pauseData), WTFMove(closestBreakpoint));
}

void PageDOMDebuggerAgent::didRemoveDOMNode(Node& node)
{
    auto nodeContainsBreakpointOwner = [&] (auto& entry) {
        return node.contains(entry.key);
    };
    m_domSubtreeModifiedBreakpoints.removeIf(nodeContainsBreakpointOwner);
    m_domAttributeModifiedBreakpoints.removeIf(nodeContainsBreakpointOwner);
    m_domNodeRemovedBreakpoints.removeIf(nodeContainsBreakpointOwner);
}

void PageDOMDebuggerAgent::willDestroyDOMNode(Node& node)
{
    // This can be called in response to GC.
    // DOM Node destruction should be treated as if the node was removed from the DOM tree.
    didRemoveDOMNode(node);
}

void PageDOMDebuggerAgent::willModifyDOMAttr(Element& element)
{
    if (!m_debuggerAgent->breakpointsActive())
        return;

    auto it = m_domAttributeModifiedBreakpoints.find(&element);
    if (it == m_domAttributeModifiedBreakpoints.end())
        return;

    auto pauseData = buildPauseDataForDOMBreakpoint(Inspector::Protocol::DOMDebugger::DOMBreakpointType::AttributeModified, element);
    m_debuggerAgent->breakProgram(Inspector::DebuggerFrontendDispatcher::Reason::DOM, WTFMove(pauseData), it->value.copyRef());
}

void PageDOMDebuggerAgent::willFireAnimationFrame()
{
    if (!m_debuggerAgent->breakpointsActive())
        return;

    auto breakpoint = m_pauseOnAllAnimationFramesBreakpoint;
    if (!breakpoint)
        return;

    m_debuggerAgent->schedulePauseForSpecialBreakpoint(*breakpoint, Inspector::DebuggerFrontendDispatcher::Reason::AnimationFrame);
}

void PageDOMDebuggerAgent::didFireAnimationFrame()
{
    if (!m_debuggerAgent->breakpointsActive())
        return;

    auto breakpoint = m_pauseOnAllAnimationFramesBreakpoint;
    if (!breakpoint)
        return;

    m_debuggerAgent->cancelPauseForSpecialBreakpoint(*breakpoint);
}

void PageDOMDebuggerAgent::willInvalidateStyleAttr(Element& element)
{
    if (!m_debuggerAgent->breakpointsActive())
        return;

    auto it = m_domAttributeModifiedBreakpoints.find(&element);
    if (it == m_domAttributeModifiedBreakpoints.end())
        return;

    auto pauseData = buildPauseDataForDOMBreakpoint(Inspector::Protocol::DOMDebugger::DOMBreakpointType::AttributeModified, element);
    m_debuggerAgent->breakProgram(Inspector::DebuggerFrontendDispatcher::Reason::DOM, WTFMove(pauseData), it->value.copyRef());
}

bool PageDOMDebuggerAgent::setAnimationFrameBreakpoint(Inspector::Protocol::ErrorString& errorString, RefPtr<JSC::Breakpoint>&& breakpoint)
{
    if (!m_pauseOnAllAnimationFramesBreakpoint == !breakpoint) {
        errorString = m_pauseOnAllAnimationFramesBreakpoint ? "Breakpoint for AnimationFrame already exists"_s : "Breakpoint for AnimationFrame missing"_s;
        return false;
    }

    m_pauseOnAllAnimationFramesBreakpoint = WTFMove(breakpoint);
    return true;
}

Ref<JSON::Object> PageDOMDebuggerAgent::buildPauseDataForDOMBreakpoint(Inspector::Protocol::DOMDebugger::DOMBreakpointType breakpointType, Node& breakpointOwner)
{
    ASSERT(m_debuggerAgent->breakpointsActive());
    ASSERT(m_domSubtreeModifiedBreakpoints.contains(&breakpointOwner) || m_domAttributeModifiedBreakpoints.contains(&breakpointOwner) || m_domNodeRemovedBreakpoints.contains(&breakpointOwner));

    auto pauseData = JSON::Object::create();
    pauseData->setString("type"_s, Inspector::Protocol::Helpers::getEnumConstantValue(breakpointType));
    if (auto* domAgent = m_instrumentingAgents.persistentDOMAgent()) {
        if (auto breakpointOwnerNodeId = domAgent->pushNodeToFrontend(&breakpointOwner))
            pauseData->setInteger("nodeId"_s, breakpointOwnerNodeId);
    }
    return pauseData;
}

} // namespace WebCore
