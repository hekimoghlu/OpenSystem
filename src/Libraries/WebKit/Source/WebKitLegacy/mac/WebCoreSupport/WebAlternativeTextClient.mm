/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 14, 2023.
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
#import "WebAlternativeTextClient.h"

#import "WebViewInternal.h"
#import <wtf/TZoneMallocInlines.h>

using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebAlternativeTextClient);

WebAlternativeTextClient::WebAlternativeTextClient(WebView* webView)
    : m_webView(webView)
{
}

WebAlternativeTextClient::~WebAlternativeTextClient()
{
#if USE(AUTOCORRECTION_PANEL)
    dismissAlternative(ReasonForDismissingAlternativeText::Ignored);
#endif
}

#if USE(AUTOCORRECTION_PANEL)
void WebAlternativeTextClient::showCorrectionAlternative(AlternativeTextType type, const FloatRect& boundingBoxOfReplacedString, const String& replacedString, const String& replacementString, const Vector<String>& alternativeReplacementStrings)
{
    m_correctionPanel.show(m_webView, type, boundingBoxOfReplacedString, replacedString, replacementString, alternativeReplacementStrings);
}

void WebAlternativeTextClient::dismissAlternative(ReasonForDismissingAlternativeText reason)
{
    m_correctionPanel.dismiss(reason);
}

String WebAlternativeTextClient::dismissAlternativeSoon(ReasonForDismissingAlternativeText reason)
{
    return m_correctionPanel.dismiss(reason);
}

static inline NSCorrectionResponse toCorrectionResponse(AutocorrectionResponse response)
{
    switch (response) {
    case WebCore::AutocorrectionResponse::Reverted:
        return NSCorrectionResponseReverted;
    case WebCore::AutocorrectionResponse::Edited:
        return NSCorrectionResponseEdited;
    case WebCore::AutocorrectionResponse::Accepted:
        return NSCorrectionResponseAccepted;
    }

    ASSERT_NOT_REACHED();
    return NSCorrectionResponseAccepted;
}

void WebAlternativeTextClient::recordAutocorrectionResponse(AutocorrectionResponse response, const String& replacedString, const String& replacementString)
{
    CorrectionPanel::recordAutocorrectionResponse([m_webView spellCheckerDocumentTag], toCorrectionResponse(response), replacedString, replacementString);
}
#endif

void WebAlternativeTextClient::removeDictationAlternatives(DictationContext dictationContext)
{
    [m_webView _removeDictationAlternatives:dictationContext];
}

void WebAlternativeTextClient::showDictationAlternativeUI(const WebCore::FloatRect& boundingBoxOfDictatedText, DictationContext dictationContext)
{
    [m_webView _showDictationAlternativeUI:boundingBoxOfDictatedText forDictationContext:dictationContext];
}

Vector<String> WebAlternativeTextClient::dictationAlternatives(DictationContext dictationContext)
{
    return [m_webView _dictationAlternatives:dictationContext];
}
