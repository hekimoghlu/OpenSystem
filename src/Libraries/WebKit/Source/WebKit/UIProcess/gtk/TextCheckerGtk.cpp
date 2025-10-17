/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 28, 2024.
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
#include "TextChecker.h"

#include "TextCheckerState.h"
#include "WebProcessPool.h"
#include <WebCore/NotImplemented.h>
#include <WebCore/TextCheckerEnchant.h>
#include <unicode/ubrk.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/text/TextBreakIterator.h>

namespace WebKit {
using namespace WebCore;

OptionSet<TextCheckerState>& checkerState()
{
    static OptionSet<TextCheckerState> textCheckerState;
    return textCheckerState;
}

OptionSet<TextCheckerState> TextChecker::state()
{
    return checkerState();
}
    
static bool testingModeEnabled = false;
    
void TextChecker::setTestingMode(bool enabled)
{
    testingModeEnabled = enabled;
}

bool TextChecker::isTestingMode()
{
    return testingModeEnabled;
}

#if ENABLE(SPELLCHECK)
static void updateStateForAllProcessPools()
{
    for (const auto& processPool : WebProcessPool::allProcessPools())
        processPool->textCheckerStateChanged();
}
#endif

bool TextChecker::isContinuousSpellCheckingAllowed()
{
#if ENABLE(SPELLCHECK)
    return true;
#else
    return false;
#endif
}

bool TextChecker::setContinuousSpellCheckingEnabled(bool isContinuousSpellCheckingEnabled)
{
#if ENABLE(SPELLCHECK)
    if (checkerState().contains(TextCheckerState::ContinuousSpellCheckingEnabled) == isContinuousSpellCheckingEnabled)
        return false;
    checkerState().set(TextCheckerState::ContinuousSpellCheckingEnabled, isContinuousSpellCheckingEnabled);
    updateStateForAllProcessPools();
#else
    UNUSED_PARAM(isContinuousSpellCheckingEnabled);
#endif
    return true;
}

void TextChecker::setGrammarCheckingEnabled(bool isGrammarCheckingEnabled)
{
#if ENABLE(SPELLCHECK)
    if (checkerState().contains(TextCheckerState::GrammarCheckingEnabled) == isGrammarCheckingEnabled)
        return;
    checkerState().set(TextCheckerState::GrammarCheckingEnabled, isGrammarCheckingEnabled);
    updateStateForAllProcessPools();
#else
    UNUSED_PARAM(isGrammarCheckingEnabled);
#endif
}

void TextChecker::continuousSpellCheckingEnabledStateChanged(bool enabled)
{
#if ENABLE(SPELLCHECK)
    checkerState().set(TextCheckerState::ContinuousSpellCheckingEnabled, enabled);
#else
    UNUSED_PARAM(enabled);
#endif
}

void TextChecker::grammarCheckingEnabledStateChanged(bool enabled)
{
#if ENABLE(SPELLCHECK)
    checkerState().set(TextCheckerState::GrammarCheckingEnabled, enabled);
#else
    UNUSED_PARAM(enabled);
#endif
}

SpellDocumentTag TextChecker::uniqueSpellDocumentTag(WebPageProxy*)
{
    return { };
}

void TextChecker::closeSpellDocumentWithTag(SpellDocumentTag)
{
}

void TextChecker::checkSpellingOfString(SpellDocumentTag, StringView text, int32_t& misspellingLocation, int32_t& misspellingLength)
{
#if ENABLE(SPELLCHECK)
    misspellingLocation = -1;
    misspellingLength = 0;
    TextCheckerEnchant::singleton().checkSpellingOfString(text.toStringWithoutCopying(), misspellingLocation, misspellingLength);
#else
    UNUSED_PARAM(text);
    UNUSED_PARAM(misspellingLocation);
    UNUSED_PARAM(misspellingLength);
#endif
}

void TextChecker::checkGrammarOfString(SpellDocumentTag, StringView /* text */, Vector<WebCore::GrammarDetail>& /* grammarDetails */, int32_t& /* badGrammarLocation */, int32_t& /* badGrammarLength */)
{
}

bool TextChecker::spellingUIIsShowing()
{
    return false;
}

void TextChecker::toggleSpellingUIIsShowing()
{
}

void TextChecker::updateSpellingUIWithMisspelledWord(SpellDocumentTag, const String& /* misspelledWord */)
{
}

void TextChecker::updateSpellingUIWithGrammarString(SpellDocumentTag, const String& /* badGrammarPhrase */, const GrammarDetail& /* grammarDetail */)
{
}

void TextChecker::getGuessesForWord(SpellDocumentTag, const String& word, const String& /* context */, int32_t /* insertionPoint */, Vector<String>& guesses, bool)
{
#if ENABLE(SPELLCHECK)
    guesses = TextCheckerEnchant::singleton().getGuessesForWord(word);
#else
    UNUSED_PARAM(word);
    UNUSED_PARAM(guesses);
#endif
}

void TextChecker::learnWord(SpellDocumentTag, const String& word)
{
#if ENABLE(SPELLCHECK)
    TextCheckerEnchant::singleton().learnWord(word);
#else
    UNUSED_PARAM(word);
#endif
}

void TextChecker::ignoreWord(SpellDocumentTag, const String& word)
{
#if ENABLE(SPELLCHECK)
    TextCheckerEnchant::singleton().ignoreWord(word);
#else
    UNUSED_PARAM(word);
#endif
}

void TextChecker::requestCheckingOfString(Ref<TextCheckerCompletion>&& completion, int32_t insertionPoint)
{
#if ENABLE(SPELLCHECK)
    TextCheckingRequestData request = completion->textCheckingRequestData();
    ASSERT(request.identifier());
    ASSERT(request.checkingTypes());

    completion->didFinishCheckingText(checkTextOfParagraph(completion->spellDocumentTag(), request.text(), insertionPoint, request.checkingTypes(), false));
#else
    UNUSED_PARAM(completion);
#endif
}

#if USE(UNIFIED_TEXT_CHECKING) && ENABLE(SPELLCHECK)
static unsigned nextWordOffset(StringView text, unsigned currentOffset)
{
    // FIXME: avoid creating textIterator object here, it could be passed as a parameter.
    //        ubrk_isBoundary() leaves the iterator pointing to the first boundary position at
    //        or after "offset" (ubrk_isBoundary side effect).
    //        For many word separators, the method doesn't properly determine the boundaries
    //        without resetting the iterator.
    UBreakIterator* textIterator = wordBreakIterator(text);
    if (!textIterator)
        return currentOffset;

    unsigned wordOffset = currentOffset;
    while (wordOffset < text.length() && ubrk_isBoundary(textIterator, wordOffset))
        ++wordOffset;

    // Do not treat the word's boundary as a separator.
    if (!currentOffset && wordOffset == 1)
        return currentOffset;

    // Omit multiple separators.
    if ((wordOffset - currentOffset) > 1)
        --wordOffset;

    return wordOffset;
}
#endif

#if USE(UNIFIED_TEXT_CHECKING)
Vector<TextCheckingResult> TextChecker::checkTextOfParagraph(SpellDocumentTag spellDocumentTag, StringView text, int32_t insertionPoint, OptionSet<TextCheckingType> checkingTypes, bool)
{
    UNUSED_PARAM(insertionPoint);
#if ENABLE(SPELLCHECK)
    if (!checkingTypes.contains(TextCheckingType::Spelling))
        return { };

    UBreakIterator* textIterator = wordBreakIterator(text);
    if (!textIterator)
        return { };

    // Omit the word separators at the beginning/end of the text to don't unnecessarily
    // involve the client to check spelling for them.
    unsigned offset = nextWordOffset(text, 0);
    unsigned lengthStrip = text.length();
    while (lengthStrip > 0 && ubrk_isBoundary(textIterator, lengthStrip - 1))
        --lengthStrip;

    Vector<TextCheckingResult> paragraphCheckingResult;
    while (offset < lengthStrip) {
        int32_t misspellingLocation = -1;
        int32_t misspellingLength = 0;
        checkSpellingOfString(spellDocumentTag, text.substring(offset, lengthStrip - offset), misspellingLocation, misspellingLength);
        if (!misspellingLength)
            break;

        TextCheckingResult misspellingResult;
        misspellingResult.type = TextCheckingType::Spelling;
        misspellingResult.range = CharacterRange(offset + misspellingLocation, misspellingLength);
        paragraphCheckingResult.append(misspellingResult);
        offset += misspellingLocation + misspellingLength;
        // Generally, we end up checking at the word separator, move to the adjacent word.
        offset = nextWordOffset(text.left(lengthStrip), offset);
    }
    return paragraphCheckingResult;
#else
    UNUSED_PARAM(spellDocumentTag);
    UNUSED_PARAM(text);
    UNUSED_PARAM(checkingTypes);
    return Vector<TextCheckingResult>();
#endif // ENABLE(SPELLCHECK)
}
#endif // USE(UNIFIED_TEXT_CHECKING)

void TextChecker::setSpellCheckingLanguages(const Vector<String>& languages)
{
#if ENABLE(SPELLCHECK)
    TextCheckerEnchant::singleton().updateSpellCheckingLanguages(languages);
#else
    UNUSED_PARAM(languages);
#endif
}

Vector<String> TextChecker::loadedSpellCheckingLanguages()
{
#if ENABLE(SPELLCHECK)
    return TextCheckerEnchant::singleton().loadedSpellCheckingLanguages();
#else
    return Vector<String>();
#endif
}

} // namespace WebKit
