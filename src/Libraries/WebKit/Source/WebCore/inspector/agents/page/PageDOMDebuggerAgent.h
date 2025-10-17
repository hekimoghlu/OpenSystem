/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 31, 2021.
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

#include "InspectorDOMDebuggerAgent.h"
#include <JavaScriptCore/Breakpoint.h>
#include <JavaScriptCore/InspectorProtocolObjects.h>
#include <wtf/RefPtr.h>

namespace WebCore {

class Element;
class LocalFrame;
class Node;

class PageDOMDebuggerAgent final : public InspectorDOMDebuggerAgent {
public:
    PageDOMDebuggerAgent(PageAgentContext&, Inspector::InspectorDebuggerAgent*);
    ~PageDOMDebuggerAgent();

    bool enabled() const;

    // DOMDebuggerBackendDispatcherHandler
    Inspector::Protocol::ErrorStringOr<void> setDOMBreakpoint(Inspector::Protocol::DOM::NodeId, Inspector::Protocol::DOMDebugger::DOMBreakpointType, RefPtr<JSON::Object>&& options);
    Inspector::Protocol::ErrorStringOr<void> removeDOMBreakpoint(Inspector::Protocol::DOM::NodeId, Inspector::Protocol::DOMDebugger::DOMBreakpointType);

    // InspectorInstrumentation
    void mainFrameNavigated();
    void frameDocumentUpdated(LocalFrame&);
    void willInsertDOMNode(Node& parent);
    void willRemoveDOMNode(Node&);
    void didRemoveDOMNode(Node&);
    void willDestroyDOMNode(Node&);
    void willModifyDOMAttr(Element&);
    void willInvalidateStyleAttr(Element&);
    void willFireAnimationFrame();
    void didFireAnimationFrame();

private:
    void enable();
    void disable();

    bool setAnimationFrameBreakpoint(Inspector::Protocol::ErrorString&, RefPtr<JSC::Breakpoint>&&);

    Ref<JSON::Object> buildPauseDataForDOMBreakpoint(Inspector::Protocol::DOMDebugger::DOMBreakpointType, Node& breakpointOwner);

    HashMap<Node*, Ref<JSC::Breakpoint>> m_domSubtreeModifiedBreakpoints;
    HashMap<Node*, Ref<JSC::Breakpoint>> m_domAttributeModifiedBreakpoints;
    HashMap<Node*, Ref<JSC::Breakpoint>> m_domNodeRemovedBreakpoints;

    RefPtr<JSC::Breakpoint> m_pauseOnAllAnimationFramesBreakpoint;
};

} // namespace WebCore
