/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 26, 2023.
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
#pragma once

#include <WebCore/AlternativeTextClient.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

class WebPage;

class WebAlternativeTextClient final : public WebCore::AlternativeTextClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(WebAlternativeTextClient);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(WebAlternativeTextClient);
public:
    explicit WebAlternativeTextClient(WebPage*);
    virtual ~WebAlternativeTextClient();

private:
#if USE(AUTOCORRECTION_PANEL)
    void showCorrectionAlternative(WebCore::AlternativeTextType, const WebCore::FloatRect& boundingBoxOfReplacedString, const String& replacedString, const String& replacementString, const Vector<String>& alternativeReplacementStrings) override;
    void dismissAlternative(WebCore::ReasonForDismissingAlternativeText) override;
    String dismissAlternativeSoon(WebCore::ReasonForDismissingAlternativeText) override;
    void recordAutocorrectionResponse(WebCore::AutocorrectionResponse, const String& replacedString, const String& replacementString) override;
#endif

#if USE(DICTATION_ALTERNATIVES)
    void showDictationAlternativeUI(const WebCore::FloatRect& boundingBoxOfDictatedText, WebCore::DictationContext) final;
    void removeDictationAlternatives(WebCore::DictationContext) final;
    Vector<String> dictationAlternatives(WebCore::DictationContext) final;
#endif

#if !(USE(AUTOCORRECTION_PANEL) || USE(DICTATION_ALTERNATIVES))
    IGNORE_CLANG_WARNINGS_BEGIN("unused-private-field")
#endif
    WeakPtr<WebPage> m_page;
#if !(USE(AUTOCORRECTION_PANEL) || USE(DICTATION_ALTERNATIVES))
    IGNORE_CLANG_WARNINGS_END
#endif
};

}
