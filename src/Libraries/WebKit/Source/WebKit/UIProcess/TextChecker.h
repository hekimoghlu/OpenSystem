/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 5, 2025.
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

#include "TextCheckerCompletion.h"
#include <WebCore/EditorClient.h>
#include <WebCore/TextCheckerClient.h>

namespace WebKit {

class WebPageProxy;
enum class TextCheckerState : uint8_t;

using SpellDocumentTag = int64_t;
    
class TextChecker {
public:
    static OptionSet<TextCheckerState> state();
    static bool isContinuousSpellCheckingAllowed();

    static bool setContinuousSpellCheckingEnabled(bool);
    static void setGrammarCheckingEnabled(bool);
    
    static void setTestingMode(bool);
    static bool isTestingMode();

#if PLATFORM(COCOA)
    static void setAutomaticSpellingCorrectionEnabled(bool);
    static void setAutomaticQuoteSubstitutionEnabled(bool);
    static void setAutomaticDashSubstitutionEnabled(bool);
    static void setAutomaticLinkDetectionEnabled(bool);
    static void setAutomaticTextReplacementEnabled(bool);

    static void didChangeAutomaticTextReplacementEnabled();
    static void didChangeAutomaticSpellingCorrectionEnabled();
    static void didChangeAutomaticQuoteSubstitutionEnabled();
    static void didChangeAutomaticDashSubstitutionEnabled();

    static bool isSmartInsertDeleteEnabled();
    static void setSmartInsertDeleteEnabled(bool);

    static bool substitutionsPanelIsShowing();
    static void toggleSubstitutionsPanelIsShowing();
#endif

#if PLATFORM(GTK)
    static void setSpellCheckingLanguages(const Vector<String>&);
    static Vector<String> loadedSpellCheckingLanguages();
#endif

    static void continuousSpellCheckingEnabledStateChanged(bool);
    static void grammarCheckingEnabledStateChanged(bool);
    
    static SpellDocumentTag uniqueSpellDocumentTag(WebPageProxy*);
    static void closeSpellDocumentWithTag(SpellDocumentTag);
#if USE(UNIFIED_TEXT_CHECKING)
    static Vector<WebCore::TextCheckingResult> checkTextOfParagraph(SpellDocumentTag, StringView, int32_t insertionPoint, OptionSet<WebCore::TextCheckingType>, bool initialCapitalizationEnabled);
#endif
    static void checkSpellingOfString(SpellDocumentTag, StringView text, int32_t& misspellingLocation, int32_t& misspellingLength);
    static void checkGrammarOfString(SpellDocumentTag, StringView text, Vector<WebCore::GrammarDetail>&, int32_t& badGrammarLocation, int32_t& badGrammarLength);
    static bool spellingUIIsShowing();
    static void toggleSpellingUIIsShowing();
    static void updateSpellingUIWithMisspelledWord(SpellDocumentTag, const String& misspelledWord);
    static void updateSpellingUIWithGrammarString(SpellDocumentTag, const String& badGrammarPhrase, const WebCore::GrammarDetail&);
    static void getGuessesForWord(SpellDocumentTag, const String& word, const String& context, int32_t insertionPoint, Vector<String>& guesses, bool initialCapitalizationEnabled);
    static void learnWord(SpellDocumentTag, const String& word);
    static void ignoreWord(SpellDocumentTag, const String& word);
    static void requestCheckingOfString(Ref<TextCheckerCompletion>&&, int32_t insertionPoint);
};

} // namespace WebKit
