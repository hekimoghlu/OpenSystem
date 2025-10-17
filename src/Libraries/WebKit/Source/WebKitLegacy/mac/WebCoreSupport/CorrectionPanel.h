/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 8, 2023.
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
#ifndef CorrectionPanel_h
#define CorrectionPanel_h

#if USE(AUTOCORRECTION_PANEL)

#import <AppKit/NSSpellChecker.h>
#import <WebCore/AlternativeTextClient.h>
#import <wtf/RetainPtr.h>

@class WebView;

class CorrectionPanel {
    WTF_MAKE_NONCOPYABLE(CorrectionPanel);
public:
    CorrectionPanel();
    ~CorrectionPanel();
    void show(WebView*, WebCore::AlternativeTextType, const WebCore::FloatRect& boundingBoxOfReplacedString, const String& replacedString, const String& replacementString, const Vector<String>& alternativeReplacementStrings);
    String dismiss(WebCore::ReasonForDismissingAlternativeText);
    static void recordAutocorrectionResponse(NSInteger spellCheckerDocumentTag, NSCorrectionResponse, const String& replacedString, const String& replacementString);

private:
    bool isShowing() const { return m_view; }
    String dismissInternal(WebCore::ReasonForDismissingAlternativeText, bool dismissingExternally);
    void handleAcceptedReplacement(NSString* acceptedReplacement, NSString* replaced, NSString* proposedReplacement, NSCorrectionIndicatorType);

    bool m_wasDismissedExternally;
    WebCore::ReasonForDismissingAlternativeText m_reasonForDismissing;
    RetainPtr<WebView> m_view;
    RetainPtr<NSString> m_resultForDismissal;
};

#endif // USE(AUTOCORRECTION_PANEL)

#endif // CorrectionPanel_h
