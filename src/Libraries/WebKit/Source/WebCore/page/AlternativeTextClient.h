/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 4, 2022.
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

#include "DictationContext.h"
#include "FloatRect.h"
#include <wtf/CheckedRef.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

enum class ReasonForDismissingAlternativeText : uint8_t {
    Cancelled = 0,
    Ignored,
    Accepted
};

enum class AlternativeTextType : uint8_t {
    Correction = 0,
    Reversion,
    SpellingSuggestions,
    GrammarSuggestions,
    DictationAlternatives
};

enum class AutocorrectionResponse : uint8_t {
    Edited,
    Reverted,
    Accepted
};

class AlternativeTextClient : public CanMakeCheckedPtr<AlternativeTextClient> {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(AlternativeTextClient);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(AlternativeTextClient);
public:
    virtual ~AlternativeTextClient() = default;
#if USE(AUTOCORRECTION_PANEL)
    virtual void showCorrectionAlternative(AlternativeTextType, const FloatRect& boundingBoxOfReplacedString, const String& replacedString, const String& replacmentString, const Vector<String>& alternativeReplacementStrings) = 0;
    virtual void dismissAlternative(ReasonForDismissingAlternativeText) = 0;
    virtual String dismissAlternativeSoon(ReasonForDismissingAlternativeText) = 0;
    virtual void recordAutocorrectionResponse(AutocorrectionResponse, const String& replacedString, const String& replacementString) = 0;
#endif
#if USE(DICTATION_ALTERNATIVES)
    virtual void showDictationAlternativeUI(const FloatRect& boundingBoxOfDictatedText, DictationContext) = 0;
    virtual void removeDictationAlternatives(DictationContext) = 0;
    virtual Vector<String> dictationAlternatives(DictationContext) = 0;
#endif
};
    
} // namespace WebCore
