/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 15, 2023.
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
#include "WebPageInspectorTarget.h"

#include "WebPage.h"
#include "WebPageInspectorTargetFrontendChannel.h"
#include <WebCore/InspectorController.h>
#include <WebCore/Page.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

namespace WebKit {

using namespace Inspector;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebPageInspectorTarget);

WebPageInspectorTarget::WebPageInspectorTarget(WebPage& page)
    : m_page(page)
{
}

WebPageInspectorTarget::~WebPageInspectorTarget() = default;

String WebPageInspectorTarget::identifier() const
{
    return toTargetID(m_page->identifier());
}

void WebPageInspectorTarget::connect(Inspector::FrontendChannel::ConnectionType connectionType)
{
    if (m_channel)
        return;
    Ref page = m_page.get();
    m_channel = makeUnique<WebPageInspectorTargetFrontendChannel>(page, identifier(), connectionType);
    if (RefPtr corePage = page->corePage())
        corePage->protectedInspectorController()->connectFrontend(*m_channel);
}

void WebPageInspectorTarget::disconnect()
{
    if (!m_channel)
        return;
    if (RefPtr corePage = m_page->corePage())
        corePage->protectedInspectorController()->disconnectFrontend(*m_channel);
    m_channel.reset();
}

void WebPageInspectorTarget::sendMessageToTargetBackend(const String& message)
{
    if (RefPtr corePage = m_page->corePage())
        corePage->protectedInspectorController()->dispatchMessageFromFrontend(message);
}

String WebPageInspectorTarget::toTargetID(WebCore::PageIdentifier pageID)
{
    return makeString("page-"_s, pageID.toUInt64());
}


} // namespace WebKit
