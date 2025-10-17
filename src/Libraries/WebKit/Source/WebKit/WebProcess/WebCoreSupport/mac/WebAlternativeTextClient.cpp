/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 27, 2023.
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
#include "WebAlternativeTextClient.h"

#include "MessageSenderInlines.h"
#include "WebPage.h"
#include "WebPageProxyMessages.h"

namespace WebKit {
using namespace WebCore;

WebAlternativeTextClient::WebAlternativeTextClient(WebPage* webPage)
    : m_page(webPage)
{
}

WebAlternativeTextClient::~WebAlternativeTextClient()
{
#if USE(AUTOCORRECTION_PANEL)
    if (m_page)
        m_page->send(Messages::WebPageProxy::DismissCorrectionPanel(ReasonForDismissingAlternativeText::Ignored));
#endif
}

#if USE(AUTOCORRECTION_PANEL)
void WebAlternativeTextClient::showCorrectionAlternative(AlternativeTextType type, const FloatRect& boundingBoxOfReplacedString, const String& replacedString, const String& replacementString, const Vector<String>& alternativeReplacementStrings)
{
    m_page->send(Messages::WebPageProxy::ShowCorrectionPanel(type, boundingBoxOfReplacedString, replacedString, replacementString, alternativeReplacementStrings));
}

void WebAlternativeTextClient::dismissAlternative(ReasonForDismissingAlternativeText reason)
{
    m_page->send(Messages::WebPageProxy::DismissCorrectionPanel(reason));
}

String WebAlternativeTextClient::dismissAlternativeSoon(ReasonForDismissingAlternativeText reason)
{
    auto sendResult = m_page->sendSync(Messages::WebPageProxy::DismissCorrectionPanelSoon(reason));
    auto [result] = sendResult.takeReplyOr(String { });
    return result;
}

void WebAlternativeTextClient::recordAutocorrectionResponse(AutocorrectionResponse response, const String& replacedString, const String& replacementString)
{
    m_page->send(Messages::WebPageProxy::RecordAutocorrectionResponse(response, replacedString, replacementString));
}
#endif

void WebAlternativeTextClient::removeDictationAlternatives(WebCore::DictationContext dictationContext)
{
    m_page->send(Messages::WebPageProxy::RemoveDictationAlternatives(dictationContext));
}

void WebAlternativeTextClient::showDictationAlternativeUI(const WebCore::FloatRect& boundingBoxOfDictatedText, WebCore::DictationContext dictationContext)
{
    m_page->send(Messages::WebPageProxy::ShowDictationAlternativeUI(boundingBoxOfDictatedText, dictationContext));
}

Vector<String> WebAlternativeTextClient::dictationAlternatives(WebCore::DictationContext dictationContext)
{
    auto sendResult = m_page->sendSync(Messages::WebPageProxy::DictationAlternatives(dictationContext));
    auto [result] = sendResult.takeReplyOr(Vector<String> { });
    return result;
}

}
