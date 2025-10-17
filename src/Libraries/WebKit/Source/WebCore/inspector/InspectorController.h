/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 25, 2024.
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

#include "InspectorOverlay.h"
#include <JavaScriptCore/InspectorAgentRegistry.h>
#include <JavaScriptCore/InspectorEnvironment.h>
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>
#include <wtf/text/WTFString.h>

namespace Inspector {
class BackendDispatcher;
class FrontendChannel;
class FrontendRouter;
class InspectorAgent;
}

namespace WebCore {

class DOMWrapperWorld;
class GraphicsContext;
class InspectorClient;
class InspectorDOMAgent;
class InspectorFrontendClient;
class InspectorInstrumentation;
class InspectorPageAgent;
class InstrumentingAgents;
class LocalFrame;
class Node;
class Page;
class PageDebugger;
class WebInjectedScriptManager;
struct PageAgentContext;

class InspectorController final : public Inspector::InspectorEnvironment, public CanMakeWeakPtr<InspectorController> {
    WTF_MAKE_NONCOPYABLE(InspectorController);
    WTF_MAKE_TZONE_ALLOCATED(InspectorController);
public:
    InspectorController(Page&, std::unique_ptr<InspectorClient>&&);
    ~InspectorController() override;

    WEBCORE_EXPORT void ref() const;
    WEBCORE_EXPORT void deref() const;

    void inspectedPageDestroyed();

    WEBCORE_EXPORT bool enabled() const;
    Page& inspectedPage() const;

    WEBCORE_EXPORT void show();

    WEBCORE_EXPORT void setInspectorFrontendClient(InspectorFrontendClient*);
    unsigned inspectionLevel() const;
    void didClearWindowObjectInWorld(LocalFrame&, DOMWrapperWorld&);

    WEBCORE_EXPORT void dispatchMessageFromFrontend(const String& message);

    bool hasLocalFrontend() const;
    bool hasRemoteFrontend() const;

    WEBCORE_EXPORT void connectFrontend(Inspector::FrontendChannel&, bool isAutomaticInspection = false, bool immediatelyPause = false);
    WEBCORE_EXPORT void disconnectFrontend(Inspector::FrontendChannel&);
    WEBCORE_EXPORT void disconnectAllFrontends();

    void inspect(Node*);
    WEBCORE_EXPORT bool shouldShowOverlay() const;
    WEBCORE_EXPORT void drawHighlight(GraphicsContext&) const;
    WEBCORE_EXPORT void getHighlight(InspectorOverlay::Highlight&, InspectorOverlay::CoordinateSystem) const;
    void hideHighlight();
    Node* highlightedNode() const;

    WEBCORE_EXPORT void setIndicating(bool);

    WEBCORE_EXPORT void willComposite(LocalFrame&);
    WEBCORE_EXPORT void didComposite(LocalFrame&);

    // Testing support.
    bool isUnderTest() const { return m_isUnderTest; }
    void setIsUnderTest(bool isUnderTest) { m_isUnderTest = isUnderTest; }
    WEBCORE_EXPORT void evaluateForTestInFrontend(const String& script);
    WEBCORE_EXPORT unsigned gridOverlayCount() const;
    WEBCORE_EXPORT unsigned flexOverlayCount() const;
    WEBCORE_EXPORT unsigned paintRectCount() const;

    InspectorClient* inspectorClient() const { return m_inspectorClient.get(); }
    InspectorFrontendClient* inspectorFrontendClient() const { return m_inspectorFrontendClient; }

    Inspector::InspectorAgent& ensureInspectorAgent();
    InspectorDOMAgent& ensureDOMAgent();
    WEBCORE_EXPORT InspectorPageAgent& ensurePageAgent();

    // InspectorEnvironment
    bool developerExtrasEnabled() const override;
    bool canAccessInspectedScriptState(JSC::JSGlobalObject*) const override;
    Inspector::InspectorFunctionCallHandler functionCallHandler() const override;
    Inspector::InspectorEvaluateHandler evaluateHandler() const override;
    void frontendInitialized() override;
    WTF::Stopwatch& executionStopwatch() const final;
    JSC::Debugger* debugger() override;
    JSC::VM& vm() override;

private:
    friend class InspectorInstrumentation;

    PageAgentContext pageAgentContext();
    void createLazyAgents();

    WeakRef<Page> m_page;
    Ref<InstrumentingAgents> m_instrumentingAgents;
    std::unique_ptr<WebInjectedScriptManager> m_injectedScriptManager;
    Ref<Inspector::FrontendRouter> m_frontendRouter;
    Ref<Inspector::BackendDispatcher> m_backendDispatcher;
    const UniqueRef<InspectorOverlay> m_overlay;
    Ref<WTF::Stopwatch> m_executionStopwatch;
    std::unique_ptr<PageDebugger> m_debugger;
    Inspector::AgentRegistry m_agents;

    std::unique_ptr<InspectorClient> m_inspectorClient;
    InspectorFrontendClient* m_inspectorFrontendClient { nullptr };

    // Lazy, but also on-demand agents.
    Inspector::InspectorAgent* m_inspectorAgent { nullptr };
    InspectorDOMAgent* m_domAgent { nullptr };
    InspectorPageAgent* m_pageAgent { nullptr };

    bool m_isUnderTest { false };
    bool m_isAutomaticInspection { false };
    bool m_pauseAfterInitialization = { false };
    bool m_didCreateLazyAgents { false };
};

} // namespace WebCore
