/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 19, 2025.
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
#import "CorrectionPanel.h"

#import "WebViewInternal.h"
#import <WebCore/CorrectionIndicator.h>
#import <wtf/cocoa/VectorCocoa.h>

#if USE(AUTOCORRECTION_PANEL)

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

void CorrectionPanel::show(WebView* view, AlternativeTextType type, const FloatRect& boundingBoxOfReplacedString, const String& replacedString, const String& replacementString, const Vector<String>& alternativeReplacementStrings)
{
    dismissInternal(ReasonForDismissingAlternativeText::Ignored, false);
    
    if (!view)
        return;

    NSString* replacedStringAsNSString = replacedString;
    NSString* replacementStringAsNSString = replacementString;
    m_view = view;
    NSCorrectionIndicatorType indicatorType = correctionIndicatorType(type);
    
    RetainPtr<NSArray> alternativeStrings;
    if (!alternativeReplacementStrings.isEmpty())
        alternativeStrings = createNSArray(alternativeReplacementStrings);

    [[NSSpellChecker sharedSpellChecker] showCorrectionIndicatorOfType:indicatorType primaryString:replacementStringAsNSString alternativeStrings:alternativeStrings.get() forStringInRect:[view _convertRectFromRootView:boundingBoxOfReplacedString] view:m_view.get() completionHandler:^(NSString* acceptedString) {
        handleAcceptedReplacement(acceptedString, replacedStringAsNSString, replacementStringAsNSString, indicatorType);
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

void CorrectionPanel::recordAutocorrectionResponse(NSInteger spellCheckerDocumentTag, NSCorrectionResponse response, const String& replacedString, const String& replacementString)
{
    [[NSSpellChecker sharedSpellChecker] recordResponse:response toCorrection:replacementString forWord:replacedString language:nil inSpellDocumentWithTag:spellCheckerDocumentTag];
}

void CorrectionPanel::handleAcceptedReplacement(NSString* acceptedReplacement, NSString* replaced, NSString* proposedReplacement,  NSCorrectionIndicatorType correctionIndicatorType)
{    
    if (!m_view)
        return;
    
    NSInteger documentTag = [m_view.get() spellCheckerDocumentTag];

    switch (correctionIndicatorType) {
    case NSCorrectionIndicatorTypeDefault:
        if (acceptedReplacement)
            recordAutocorrectionResponse(documentTag, NSCorrectionResponseAccepted, replaced, acceptedReplacement);
        else {
            if (!m_wasDismissedExternally || m_reasonForDismissing == ReasonForDismissingAlternativeText::Cancelled)
                recordAutocorrectionResponse(documentTag, NSCorrectionResponseRejected, replaced, proposedReplacement);
            else
                recordAutocorrectionResponse(documentTag, NSCorrectionResponseIgnored, replaced, proposedReplacement);
        }
        break;
    case NSCorrectionIndicatorTypeReversion:
        if (acceptedReplacement)
            recordAutocorrectionResponse(documentTag, NSCorrectionResponseReverted, replaced, acceptedReplacement);
        break;
    case NSCorrectionIndicatorTypeGuesses:
        if (acceptedReplacement)
            recordAutocorrectionResponse(documentTag, NSCorrectionResponseAccepted, replaced, acceptedReplacement);
        break;
    }
    
    [m_view.get() handleAcceptedAlternativeText:acceptedReplacement];
    m_view.clear();
    if (acceptedReplacement)
        m_resultForDismissal = adoptNS([acceptedReplacement copy]);
}

#endif //USE(AUTOCORRECTION_PANEL)

