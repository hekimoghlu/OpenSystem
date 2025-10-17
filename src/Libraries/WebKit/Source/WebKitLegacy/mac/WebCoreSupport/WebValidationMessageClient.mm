/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 22, 2021.
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
#import "WebValidationMessageClient.h"

#import "WebView.h"
#import "WebViewInternal.h"
#import <WebCore/Element.h>
#import <wtf/TZoneMallocInlines.h>

using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebValidationMessageClient);

WebValidationMessageClient::WebValidationMessageClient(WebView* view)
    : m_view(view)
{
}

WebValidationMessageClient::~WebValidationMessageClient()
{
    if (m_currentAnchor)
        hideValidationMessage(*m_currentAnchor);
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

    m_currentAnchor = &anchor;
    m_currentAnchorRect = anchor.boundingBoxInRootViewCoordinates();
    [m_view showFormValidationMessage:message withAnchorRect:m_currentAnchorRect];
}

void WebValidationMessageClient::hideValidationMessage(const Element& anchor)
{
    if (!isValidationMessageVisible(anchor))
        return;

    m_currentAnchor = nullptr;
    m_currentAnchorRect = { };
    [m_view hideFormValidationMessage];
}

void WebValidationMessageClient::hideAnyValidationMessage()
{
    if (!m_currentAnchor)
        return;

    m_currentAnchor = nullptr;
    m_currentAnchorRect = { };
    [m_view hideFormValidationMessage];
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
