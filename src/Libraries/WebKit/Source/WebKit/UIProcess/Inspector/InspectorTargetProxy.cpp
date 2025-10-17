/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 30, 2023.
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
#include "InspectorTargetProxy.h"

#include "MessageSenderInlines.h"
#include "ProvisionalPageProxy.h"
#include "WebFrameProxy.h"
#include "WebPageInspectorTarget.h"
#include "WebPageMessages.h"
#include "WebPageProxy.h"
#include "WebProcessProxy.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

using namespace Inspector;

WTF_MAKE_TZONE_ALLOCATED_IMPL(InspectorTargetProxy);

std::unique_ptr<InspectorTargetProxy> InspectorTargetProxy::create(WebPageProxy& page, const String& targetId, Inspector::InspectorTargetType type)
{
    return makeUnique<InspectorTargetProxy>(page, targetId, type);
}

std::unique_ptr<InspectorTargetProxy> InspectorTargetProxy::create(ProvisionalPageProxy& provisionalPage, const String& targetId, Inspector::InspectorTargetType type)
{
    RefPtr page = provisionalPage.page();
    if (!page)
        return nullptr;

    auto target = InspectorTargetProxy::create(*page, targetId, type);
    target->m_provisionalPage = provisionalPage;
    return target;
}

InspectorTargetProxy::InspectorTargetProxy(WebPageProxy& page, const String& targetId, Inspector::InspectorTargetType type)
    : m_page(page)
    , m_identifier(targetId)
    , m_type(type)
{
}

void InspectorTargetProxy::connect(Inspector::FrontendChannel::ConnectionType connectionType)
{
    if (RefPtr provisionalPage = m_provisionalPage.get()) {
        provisionalPage->send(Messages::WebPage::ConnectInspector(identifier(), connectionType));
        return;
    }

    if (m_page->hasRunningProcess())
        m_page->legacyMainFrameProcess().send(Messages::WebPage::ConnectInspector(identifier(), connectionType), m_page->webPageIDInMainFrameProcess());
}

void InspectorTargetProxy::disconnect()
{
    if (isPaused())
        resume();

    if (RefPtr provisionalPage = m_provisionalPage.get()) {
        provisionalPage->send(Messages::WebPage::DisconnectInspector(identifier()));
        return;
    }

    if (m_page->hasRunningProcess())
        m_page->legacyMainFrameProcess().send(Messages::WebPage::DisconnectInspector(identifier()), m_page->webPageIDInMainFrameProcess());
}

void InspectorTargetProxy::sendMessageToTargetBackend(const String& message)
{
    if (RefPtr provisionalPage = m_provisionalPage.get()) {
        provisionalPage->send(Messages::WebPage::SendMessageToTargetBackend(identifier(), message));
        return;
    }

    if (m_page->hasRunningProcess())
        m_page->legacyMainFrameProcess().send(Messages::WebPage::SendMessageToTargetBackend(identifier(), message), m_page->webPageIDInMainFrameProcess());
}

void InspectorTargetProxy::didCommitProvisionalTarget()
{
    m_provisionalPage = nullptr;
}

bool InspectorTargetProxy::isProvisional() const
{
    return !!m_provisionalPage;
}

} // namespace WebKit
