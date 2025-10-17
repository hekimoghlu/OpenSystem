/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 10, 2021.
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

#if !PLATFORM(COCOA) && !PLATFORM(GTK)
#include "TextCheckerState.h"
#include <WebCore/NotImplemented.h>

namespace WebKit {
using namespace WebCore;

static OptionSet<TextCheckerState>& checkerState()
{
    static OptionSet<TextCheckerState> textCheckerState;
    return textCheckerState;
}

OptionSet<TextCheckerState> TextChecker::state()
{
    return checkerState();
}

void TextChecker::setTestingMode(bool)
{
}

bool TextChecker::isTestingMode()
{
    notImplemented();
    return false;
}

bool TextChecker::isContinuousSpellCheckingAllowed()
{
    notImplemented();
    return false;
}

bool TextChecker::setContinuousSpellCheckingEnabled(bool)
{
    notImplemented();
    return false;
}

void TextChecker::setGrammarCheckingEnabled(bool)
{
    notImplemented();
}

void TextChecker::continuousSpellCheckingEnabledStateChanged(bool)
{
    notImplemented();
}

void TextChecker::grammarCheckingEnabledStateChanged(bool)
{
    notImplemented();
}

SpellDocumentTag TextChecker::uniqueSpellDocumentTag(WebPageProxy*)
{
    notImplemented();
    return false;
}

void TextChecker::closeSpellDocumentWithTag(SpellDocumentTag)
{
    notImplemented();
}

void TextChecker::checkSpellingOfString(SpellDocumentTag, StringView, int32_t&, int32_t&)
{
    notImplemented();
}

void TextChecker::checkGrammarOfString(SpellDocumentTag, StringView, Vector<WebCore::GrammarDetail>&, int32_t&, int32_t&)
{
    notImplemented();
}

bool TextChecker::spellingUIIsShowing()
{
    notImplemented();
    return false;
}

void TextChecker::toggleSpellingUIIsShowing()
{
    notImplemented();
}

void TextChecker::updateSpellingUIWithMisspelledWord(SpellDocumentTag, const String&)
{
    notImplemented();
}

void TextChecker::updateSpellingUIWithGrammarString(SpellDocumentTag, const String&, const GrammarDetail&)
{
    notImplemented();
}

void TextChecker::getGuessesForWord(SpellDocumentTag, const String&, const String&, int32_t, Vector<String>&, bool)
{
    notImplemented();
}

void TextChecker::learnWord(SpellDocumentTag, const String&)
{
    notImplemented();
}

void TextChecker::ignoreWord(SpellDocumentTag, const String&)
{
    notImplemented();
}

void TextChecker::requestCheckingOfString(Ref<TextCheckerCompletion>&&, int32_t)
{
    notImplemented();
}

#if USE(UNIFIED_TEXT_CHECKING)
Vector<TextCheckingResult> TextChecker::checkTextOfParagraph(SpellDocumentTag, StringView, int32_t, OptionSet<TextCheckingType>, bool)
{
    notImplemented();
    return { };
}
#endif

} // namespace WebKit 

#endif // !PLATFORM(COCOA) && !PLATFORM(GTK)
