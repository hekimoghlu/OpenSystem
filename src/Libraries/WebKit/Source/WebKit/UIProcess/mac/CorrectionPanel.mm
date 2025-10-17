/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 16, 2025.
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
#import "config.h"
#import "CorrectionPanel.h"

#if USE(AUTOCORRECTION_PANEL)

#import "WebPageProxy.h"
#import "WebViewImpl.h"
#import <WebCore/CorrectionIndicator.h>
#import <pal/SessionID.h>
#import <wtf/cocoa/VectorCocoa.h>

namespace WebKit {
using namespace WebCore;

CorrectionPanel::CorrectionPanel()
    : m_wasDismissedExternally(false)
    , m_reasonForDismissing(ReasonForDismissingAlternativeText::Ignored)
{
}

CorrectionPanel::~CorrectionPanel()
{
    dismissInternal(ReasonForDismissingAlternativeText::Ignored, false);
}

void CorrectionPanel::show(NSView *view, WebViewImpl& webViewImpl, AlternativeTextType type, const FloatRect& boundingBoxOfReplacedString, const String& replacedString, const String& replacementString, const Vector<String>& alternativeReplacementStrings)
{
    dismissInternal(ReasonForDismissingAlternativeText::Ignored, false);

    if (!view)
        return;

    NSInteger spellCheckerDocumentTag = webViewImpl.spellCheckerDocumentTag();

    NSString *replacedStringAsNSString = replacedString;
    NSString *replacementStringAsNSString = replacementString;

    m_view = view;
    m_spellCheckerDocumentTag = spellCheckerDocumentTag;
    NSCorrectionIndicatorType indicatorType = correctionIndicatorType(type);

    RetainPtr<NSArray> alternativeStrings;
    if (!alternativeReplacementStrings.isEmpty())
        alternativeStrings = createNSArray(alternativeReplacementStrings);

    NSSpellChecker* spellChecker = [NSSpellChecker sharedSpellChecker];
    [spellChecker showCorrectionIndicatorOfType:indicatorType primaryString:replacementStringAsNSString alternativeStrings:alternativeStrings.get() forStringInRect:boundingBoxOfReplacedString view:m_view.get() completionHandler:^(NSString* acceptedString) {
        handleAcceptedReplacement(webViewImpl, acceptedString, replacedStringAsNSString, replacementStringAsNSString, indicatorType);
    }];
}

String CorrectionPanel::dismiss(ReasonForDismissingAlternativeText reason)
{
    return dismissInternal(reason, true);
}

String CorrectionPanel::dismissInternal(ReasonForDismissingAlternativeText reason, bool dismissingExternally)
{
    if (!isShowing())
        return String();

    m_wasDismissedExternally = dismissingExternally;
    m_reasonForDismissing = reason;
    m_resultForDismissal.clear();
    [[NSSpellChecker sharedSpellChecker] dismissCorrectionIndicatorForView:m_view.get()];
    return m_resultForDismissal.get();
}

void CorrectionPanel::recordAutocorrectionResponse(WebViewImpl& webViewImpl, NSInteger spellCheckerDocumentTag, NSCorrectionResponse response, const String& replacedString, const String& replacementString)
{
    if (webViewImpl.page().sessionID().isEphemeral())
        return;

    [[NSSpellChecker sharedSpellChecker] recordResponse:response toCorrection:replacementString forWord:replacedString language:nil inSpellDocumentWithTag:spellCheckerDocumentTag];
}

void CorrectionPanel::handleAcceptedReplacement(WebViewImpl& webViewImpl, NSString* acceptedReplacement, NSString* replaced, NSString* proposedReplacement,  NSCorrectionIndicatorType correctionIndicatorType)
{
    if (!m_view)
        return;

    switch (correctionIndicatorType) {
    case NSCorrectionIndicatorTypeDefault:
        if (acceptedReplacement)
            recordAutocorrectionResponse(webViewImpl, m_spellCheckerDocumentTag, NSCorrectionResponseAccepted, replaced, acceptedReplacement);
        else {
            if (!m_wasDismissedExternally || m_reasonForDismissing == ReasonForDismissingAlternativeText::Cancelled)
                recordAutocorrectionResponse(webViewImpl, m_spellCheckerDocumentTag, NSCorrectionResponseRejected, replaced, proposedReplacement);
            else
                recordAutocorrectionResponse(webViewImpl, m_spellCheckerDocumentTag, NSCorrectionResponseIgnored, replaced, proposedReplacement);
        }
        break;
    case NSCorrectionIndicatorTypeReversion:
        if (acceptedReplacement)
            recordAutocorrectionResponse(webViewImpl, m_spellCheckerDocumentTag, NSCorrectionResponseReverted, replaced, acceptedReplacement);
        break;
    case NSCorrectionIndicatorTypeGuesses:
        if (acceptedReplacement)
            recordAutocorrectionResponse(webViewImpl, m_spellCheckerDocumentTag, NSCorrectionResponseAccepted, replaced, acceptedReplacement);
        break;
    }

    webViewImpl.handleAcceptedAlternativeText(acceptedReplacement);
    m_spellCheckerDocumentTag = 0;
    m_view = nullptr;
    if (acceptedReplacement)
        m_resultForDismissal = adoptNS([acceptedReplacement copy]);
}

} // namespace WebKit

#endif // USE(AUTOCORRECTION_PANEL)
