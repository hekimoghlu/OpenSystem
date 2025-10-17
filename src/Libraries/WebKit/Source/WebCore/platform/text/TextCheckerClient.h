/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 4, 2023.
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

#include "TextChecking.h"

namespace WebCore {

class VisibleSelection;

class TextCheckerClient {
public:
    virtual ~TextCheckerClient() = default;

    virtual bool shouldEraseMarkersAfterChangeSelection(TextCheckingType) const = 0;
    virtual void ignoreWordInSpellDocument(const String&) = 0;
    virtual void learnWord(const String&) = 0;
    virtual void checkSpellingOfString(StringView, int* misspellingLocation, int* misspellingLength) = 0;
    virtual void checkGrammarOfString(StringView, Vector<GrammarDetail>&, int* badGrammarLocation, int* badGrammarLength) = 0;

#if USE(UNIFIED_TEXT_CHECKING)
    virtual Vector<TextCheckingResult> checkTextOfParagraph(StringView, OptionSet<TextCheckingType> checkingTypes, const VisibleSelection& currentSelection) = 0;
#endif

    // For spellcheckers that support multiple languages, it's often important to be able to identify the language in order to
    // provide more accurate correction suggestions. Caller can pass in more text in "context" to aid such spellcheckers on language
    // identification. Noramlly it's the text surrounding the "word" for which we are getting correction suggestions.
    virtual void getGuessesForWord(const String& word, const String& context, const VisibleSelection& currentSelection, Vector<String>& guesses) = 0;
    virtual void requestCheckingOfString(TextCheckingRequest&, const VisibleSelection& currentSelection) = 0;
};

}
