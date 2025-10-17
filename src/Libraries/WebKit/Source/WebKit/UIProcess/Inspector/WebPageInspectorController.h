/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 18, 2022.
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

#include "InspectorTargetProxy.h"
#include <JavaScriptCore/InspectorAgentRegistry.h>
#include <JavaScriptCore/InspectorTargetAgent.h>
#include <WebCore/PageIdentifier.h>
#include <wtf/CheckedRef.h>
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace Inspector {
class BackendDispatcher;
class FrontendChannel;
class FrontendRouter;
}

namespace WebKit {

class InspectorBrowserAgent;
struct WebPageAgentContext;

class WebPageInspectorController {
    WTF_MAKE_TZONE_ALLOCATED(WebPageInspectorController);
    WTF_MAKE_NONCOPYABLE(WebPageInspectorController);
public:
    WebPageInspectorController(WebPageProxy&);
    ~WebPageInspectorController();

    void init();
    void pageClosed();

    bool hasLocalFrontend() const;

    void connectFrontend(Inspector::FrontendChannel&, bool isAutomaticInspection = false, bool immediatelyPause = false);
    void disconnectFrontend(Inspector::FrontendChannel&);
    void disconnectAllFrontends();

    void dispatchMessageFromFrontend(const String& message);

#if ENABLE(REMOTE_INSPECTOR)
    void setIndicating(bool);
#endif

    void createInspectorTarget(const String& targetId, Inspector::InspectorTargetType);
    void destroyInspectorTarget(const String& targetId);
    void sendMessageToInspectorFrontend(const String& targetId, const String& message);

    bool shouldPauseLoading(const ProvisionalPageProxy&) const;
    void setContinueLoadingCallback(const ProvisionalPageProxy&, WTF::Function<void()>&&);

    void didCreateProvisionalPage(ProvisionalPageProxy&);
    void willDestroyProvisionalPage(const ProvisionalPageProxy&);
    void didCommitProvisionalPage(WebCore::PageIdentifier oldWebPageID, WebCore::PageIdentifier newWebPageID);

    InspectorBrowserAgent* enabledBrowserAgent() const;
    void setEnabledBrowserAgent(InspectorBrowserAgent*);

    void browserExtensionsEnabled(HashMap<String, String>&&);
    void browserExtensionsDisabled(HashSet<String>&&);

private:
    Ref<WebPageProxy> protectedInspectedPage();
    WebPageAgentContext webPageAgentContext();
    void createLazyAgents();

    void addTarget(std::unique_ptr<InspectorTargetProxy>&&);

    Ref<Inspector::FrontendRouter> m_frontendRouter;
    Ref<Inspector::BackendDispatcher> m_backendDispatcher;
    Inspector::AgentRegistry m_agents;

    WeakRef<WebPageProxy> m_inspectedPage;

    CheckedPtr<Inspector::InspectorTargetAgent> m_targetAgent;
    HashMap<String, std::unique_ptr<InspectorTargetProxy>> m_targets;

    CheckedPtr<InspectorBrowserAgent> m_enabledBrowserAgent;

    bool m_didCreateLazyAgents { false };
};

} // namespace WebKit
