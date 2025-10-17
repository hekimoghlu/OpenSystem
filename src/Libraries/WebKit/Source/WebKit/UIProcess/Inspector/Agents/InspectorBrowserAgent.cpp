/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 15, 2021.
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
#include "InspectorBrowserAgent.h"

#include "APIUIClient.h"
#include "WebInspectorUIProxy.h"
#include "WebPageInspectorController.h"
#include "WebPageProxy.h"
#include <JavaScriptCore/InspectorProtocolObjects.h>
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/WTFString.h>

namespace WebKit {

using namespace Inspector;

WTF_MAKE_TZONE_ALLOCATED_IMPL(InspectorBrowserAgent);

InspectorBrowserAgent::InspectorBrowserAgent(WebPageAgentContext& context)
    : InspectorAgentBase("Browser"_s, context)
    , m_frontendDispatcher(makeUnique<Inspector::BrowserFrontendDispatcher>(context.frontendRouter))
    , m_backendDispatcher(Inspector::BrowserBackendDispatcher::create(context.backendDispatcher, this))
    , m_inspectedPage(context.inspectedPage)
{
}

InspectorBrowserAgent::~InspectorBrowserAgent() = default;

bool InspectorBrowserAgent::enabled() const
{
    return m_inspectedPage->inspectorController().enabledBrowserAgent() == this;
}

void InspectorBrowserAgent::didCreateFrontendAndBackend(Inspector::FrontendRouter*, Inspector::BackendDispatcher*)
{
}

void InspectorBrowserAgent::willDestroyFrontendAndBackend(Inspector::DisconnectReason)
{
    disable();
}

Inspector::Protocol::ErrorStringOr<void> InspectorBrowserAgent::enable()
{
    if (enabled())
        return makeUnexpected("Browser domain already enabled"_s);

    m_inspectedPage->inspectorController().setEnabledBrowserAgent(this);

    return { };
}

Inspector::Protocol::ErrorStringOr<void> InspectorBrowserAgent::disable()
{
    if (!enabled())
        return makeUnexpected("Browser domain already disabled"_s);

    m_inspectedPage->inspectorController().setEnabledBrowserAgent(nullptr);

    return { };
}

void InspectorBrowserAgent::extensionsEnabled(HashMap<String, String>&& extensionIDToName)
{
    ASSERT(enabled());

    auto extensionsPayload = JSON::ArrayOf<Inspector::Protocol::Browser::Extension>::create();
    for (auto& [id, name] : extensionIDToName) {
        auto extensionPayload = Inspector::Protocol::Browser::Extension::create()
            .setExtensionId(id)
            .setName(name)
            .release();
        extensionsPayload->addItem(WTFMove(extensionPayload));
    }
    m_frontendDispatcher->extensionsEnabled(WTFMove(extensionsPayload));
}

void InspectorBrowserAgent::extensionsDisabled(HashSet<String>&& extensionIDs)
{
    ASSERT(enabled());

    auto extensionIdsPayload = JSON::ArrayOf<String>::create();
    for (auto& extensionId : extensionIDs)
        extensionIdsPayload->addItem(extensionId);
    m_frontendDispatcher->extensionsDisabled(WTFMove(extensionIdsPayload));
}


} // namespace WebCore
