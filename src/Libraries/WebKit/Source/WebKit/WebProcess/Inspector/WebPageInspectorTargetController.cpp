/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 7, 2025.
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
#include "WebPageInspectorTargetController.h"

#include "MessageSenderInlines.h"
#include "WebPage.h"
#include "WebPageInspectorTargetFrontendChannel.h"
#include "WebPageProxyMessages.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebPageInspectorTargetController);

WebPageInspectorTargetController::WebPageInspectorTargetController(WebPage& page)
    : m_page(page)
    , m_pageTarget(page)
{
    // Do not send the page target to the UIProcess, the WebPageProxy will manager this for us.
    m_targets.set(m_pageTarget.identifier(), &m_pageTarget);
}

WebPageInspectorTargetController::~WebPageInspectorTargetController() = default;

Ref<WebPage> WebPageInspectorTargetController::protectedPage() const
{
    return m_page.get();
}

void WebPageInspectorTargetController::addTarget(Inspector::InspectorTarget& target)
{
    auto addResult = m_targets.set(target.identifier(), &target);
    ASSERT_UNUSED(addResult, addResult.isNewEntry);

    protectedPage()->send(Messages::WebPageProxy::CreateInspectorTarget(target.identifier(), target.type()));
}

void WebPageInspectorTargetController::removeTarget(Inspector::InspectorTarget& target)
{
    ASSERT_WITH_MESSAGE(target.identifier() != m_pageTarget.identifier(), "Should never remove the main target.");

    protectedPage()->send(Messages::WebPageProxy::DestroyInspectorTarget(target.identifier()));

    m_targets.remove(target.identifier());
}

void WebPageInspectorTargetController::connectInspector(const String& targetId, Inspector::FrontendChannel::ConnectionType connectionType)
{
    auto target = m_targets.get(targetId);
    if (!target)
        return;

    target->connect(connectionType);
}

void WebPageInspectorTargetController::disconnectInspector(const String& targetId)
{
    auto target = m_targets.get(targetId);
    if (!target)
        return;

    target->disconnect();
}

void WebPageInspectorTargetController::sendMessageToTargetBackend(const String& targetId, const String& message)
{
    auto target = m_targets.get(targetId);
    if (!target)
        return;

    target->sendMessageToTargetBackend(message);
}

} // namespace WebKit
