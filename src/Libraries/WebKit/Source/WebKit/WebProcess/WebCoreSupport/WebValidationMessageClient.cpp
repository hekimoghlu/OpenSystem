/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 1, 2022.
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
#include "WebValidationMessageClient.h"

#include "MessageSenderInlines.h"
#include "WebPage.h"
#include "WebPageProxyMessages.h"
#include <WebCore/Element.h>
#include <WebCore/LocalFrame.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebValidationMessageClient);

WebValidationMessageClient::WebValidationMessageClient(WebPage& page)
    : m_page(page)
{
}

WebValidationMessageClient::~WebValidationMessageClient()
{
    if (RefPtr anchor = m_currentAnchor.get())
        hideValidationMessage(*anchor);
}

void WebValidationMessageClient::documentDetached(Document& document)
{
    if (!m_currentAnchor)
        return;
    if (&m_currentAnchor->document() == &document)
        hideValidationMessage(*m_currentAnchor);
}

void WebValidationMessageClient::showValidationMessage(const Element& anchor, const String& message)
{
    if (m_currentAnchor)
        hideValidationMessage(*m_currentAnchor);

    m_currentAnchor = anchor;
    m_currentAnchorRect = anchor.boundingBoxInRootViewCoordinates();
    Ref { *m_page }->send(Messages::WebPageProxy::ShowValidationMessage(m_currentAnchorRect, message));
}

void WebValidationMessageClient::hideValidationMessage(const Element& anchor)
{
    RefPtr page = m_page.get();
    if (!isValidationMessageVisible(anchor) || !page)
        return;

    m_currentAnchor = nullptr;
    m_currentAnchorRect = { };
    page->send(Messages::WebPageProxy::HideValidationMessage());
}

void WebValidationMessageClient::hideAnyValidationMessage()
{
    RefPtr page = m_page.get();
    if (!m_currentAnchor || !page)
        return;

    m_currentAnchor = nullptr;
    m_currentAnchorRect = { };
    page->send(Messages::WebPageProxy::HideValidationMessage());
}

bool WebValidationMessageClient::isValidationMessageVisible(const Element& anchor)
{
    return m_currentAnchor == &anchor;
}

void WebValidationMessageClient::updateValidationBubbleStateIfNeeded()
{
    if (!m_currentAnchor)
        return;

    // We currently hide the validation bubble if its position is outdated instead of trying
    // to update its position.
    if (m_currentAnchorRect != m_currentAnchor->boundingBoxInRootViewCoordinates())
        hideValidationMessage(*m_currentAnchor);
}

} // namespace WebKit
