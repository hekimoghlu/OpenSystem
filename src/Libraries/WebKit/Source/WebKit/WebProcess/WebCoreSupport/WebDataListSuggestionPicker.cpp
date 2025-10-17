/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 13, 2023.
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
#include "WebDataListSuggestionPicker.h"

#include "WebPage.h"
#include "WebPageProxyMessages.h"
#include "WebProcess.h"
#include <WebCore/DataListSuggestionsClient.h>
#include <WebCore/LocalFrameView.h>

namespace WebKit {

WebDataListSuggestionPicker::WebDataListSuggestionPicker(WebPage& page, WebCore::DataListSuggestionsClient& client)
    : m_client(client)
    , m_page(page)
{
}

WebDataListSuggestionPicker::~WebDataListSuggestionPicker() = default;

void WebDataListSuggestionPicker::handleKeydownWithIdentifier(const String& key)
{
    if (key == "U+001B"_s) {
        close();
        return;
    }

    RefPtr page = m_page.get();
    if (!page)
        return;

    WebProcess::singleton().parentProcessConnection()->send(Messages::WebPageProxy::HandleKeydownInDataList(key), page->identifier());
}

void WebDataListSuggestionPicker::didSelectOption(const String& selectedOption)
{
    if (CheckedPtr client = m_client)
        client->didSelectDataListOption(selectedOption);
}

void WebDataListSuggestionPicker::didCloseSuggestions()
{
    if (CheckedPtr client = m_client)
        client->didCloseSuggestions();
}

void WebDataListSuggestionPicker::close()
{
    RefPtr page = m_page.get();
    if (!page)
        return;

    WebProcess::singleton().parentProcessConnection()->send(Messages::WebPageProxy::EndDataListSuggestions(), page->identifier());
}

void WebDataListSuggestionPicker::displayWithActivationType(WebCore::DataListSuggestionActivationType type)
{
    CheckedPtr client = m_client;
    if (!client)
        return;

    auto suggestions = client->suggestions();
    if (suggestions.isEmpty()) {
        close();
        return;
    }

    RefPtr page = m_page.get();
    if (!page)
        return;

    auto elementRectInRootViewCoordinates = client->elementRectInRootViewCoordinates();
    if (RefPtr view = page->localMainFrameView()) {
        auto unobscuredRootViewRect = view->contentsToRootView(view->unobscuredContentRect());
        if (!unobscuredRootViewRect.intersects(elementRectInRootViewCoordinates))
            return close();
    }

    page->setActiveDataListSuggestionPicker(*this);

    WebCore::DataListSuggestionInformation info { type, WTFMove(suggestions), WTFMove(elementRectInRootViewCoordinates) };
    WebProcess::singleton().parentProcessConnection()->send(Messages::WebPageProxy::ShowDataListSuggestions(info), page->identifier());
}

void WebDataListSuggestionPicker::detach()
{
    m_client = nullptr;
}

} // namespace WebKit
