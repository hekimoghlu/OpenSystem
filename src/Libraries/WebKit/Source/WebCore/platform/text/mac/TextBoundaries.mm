/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 29, 2024.
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
#import "TextBoundaries.h"

#import <CoreFoundation/CFStringTokenizer.h>
#import <Foundation/Foundation.h>
#import <unicode/ubrk.h>
#import <unicode/uchar.h>
#import <unicode/ustring.h>
#import <unicode/utypes.h>
#import <wtf/RetainPtr.h>
#import <wtf/text/StringView.h>
#import <wtf/text/TextBreakIterator.h>
#import <wtf/text/TextBreakIteratorInternalICU.h>
#import <wtf/unicode/CharacterNames.h>

namespace WebCore {

#if !USE(APPKIT)

static bool isWordDelimitingCharacter(char32_t c)
{
    // Ampersand is an exception added to treat AT&T as a single word (see <rdar://problem/5022264>).
    return !CFCharacterSetIsLongCharacterMember(CFCharacterSetGetPredefined(kCFCharacterSetAlphaNumeric), c) && c != '&';
}

static bool isSymbolCharacter(char32_t c)
{
    return CFCharacterSetIsLongCharacterMember(CFCharacterSetGetPredefined(kCFCharacterSetSymbol), c);
}

static bool isAmbiguousBoundaryCharacter(char32_t character)
{
    // These are characters that can behave as word boundaries, but can appear within words.
    return character == '\'' || character == rightSingleQuotationMark || character == hebrewPunctuationGershayim;
}

static CFStringTokenizerRef tokenizerForString(CFStringRef str)
{
    static const NeverDestroyed locale = [] {
        const char* localID = currentTextBreakLocaleID();
        auto currentLocaleID = adoptCF(CFStringCreateWithBytesNoCopy(kCFAllocatorDefault, byteCast<UInt8>(localID), strlen(localID), kCFStringEncodingASCII, false, kCFAllocatorNull));
        return adoptCF(CFLocaleCreate(kCFAllocatorDefault, currentLocaleID.get()));
    }();

    if (!locale.get())
        return nullptr;

    CFRange entireRange = CFRangeMake(0, CFStringGetLength(str));    

    static NeverDestroyed<RetainPtr<CFStringTokenizerRef>> tokenizer;
    if (!tokenizer.get())
        tokenizer.get() = adoptCF(CFStringTokenizerCreate(kCFAllocatorDefault, str, entireRange, kCFStringTokenizerUnitWordBoundary, locale.get().get()));
    else
        CFStringTokenizerSetString(tokenizer.get().get(), str, entireRange);
    return tokenizer.get().get();
}

// Simple case: A word is a stream of characters delimited by a special set of word-delimiting characters.
static void findSimpleWordBoundary(StringView text, int position, int* start, int* end)
{
    ASSERT(position >= 0);
    ASSERT(static_cast<unsigned>(position) < text.length());

    unsigned startPos = position;
    while (startPos > 0) {
        int i = startPos;
        char32_t characterBeforeStartPos;
        U16_PREV(text, 0, i, characterBeforeStartPos);
        if (isWordDelimitingCharacter(characterBeforeStartPos)) {
            ASSERT(i >= 0);
            if (!i)
                break;

            if (!isAmbiguousBoundaryCharacter(characterBeforeStartPos))
                break;

            char32_t characterBeforeBeforeStartPos;
            U16_PREV(text, 0, i, characterBeforeBeforeStartPos);
            if (isWordDelimitingCharacter(characterBeforeBeforeStartPos))
                break;
        }
        U16_BACK_1(text, 0, startPos);
    }
    
    unsigned endPos = position;
    while (endPos < text.length()) {
        char32_t character;
        U16_GET(text, 0, endPos, text.length(), character);
        if (isWordDelimitingCharacter(character)) {
            unsigned i = endPos;
            U16_FWD_1(text, i, text.length());
            ASSERT(i <= text.length());
            if (i == text.length())
                break;
            char32_t characterAfterEndPos;
            U16_NEXT(text, i, text.length(), characterAfterEndPos);
            if (!isAmbiguousBoundaryCharacter(character))
                break;
            if (isWordDelimitingCharacter(characterAfterEndPos))
                break;
        }
        U16_FWD_1(text, endPos, text.length());
    }

    // The text may consist of all delimiter characters (e.g. "++++++++" or a series of emoji), and returning an empty range
    // makes no sense (and doesn't match findComplexWordBoundary() behavior).
    if (startPos == endPos && endPos < text.length()) {
        char32_t character;
        U16_GET(text, 0, endPos, text.length(), character);
        if (isSymbolCharacter(character))
            U16_FWD_1(text, endPos, text.length());
    }

    *start = startPos;
    *end = endPos;
}

// Complex case: use CFStringTokenizer to find word boundary.
static void findComplexWordBoundary(StringView text, int position, int* start, int* end)
{
    RetainPtr<CFStringRef> charString = text.createCFStringWithoutCopying();

    CFStringTokenizerRef tokenizer = tokenizerForString(charString.get());
    if (!tokenizer) {
        // Error creating tokenizer, so just use simple function.
        findSimpleWordBoundary(text, position, start, end);
        return;
    }

    CFStringTokenizerTokenType  token = CFStringTokenizerGoToTokenAtIndex(tokenizer, position);
    if (token == kCFStringTokenizerTokenNone) {
        // No token found: select entire block.
        // NB: I never hit this section in all my testing.
        *start = 0;
        *end = text.length();
        return;
    }

    CFRange result = CFStringTokenizerGetCurrentTokenRange(tokenizer);
    *start = result.location;
    *end = result.location + result.length;
}

#endif

void findWordBoundary(StringView text, int position, int* start, int* end)
{
#if USE(APPKIT)
    auto attributedString = adoptNS([[NSAttributedString alloc] initWithString:text.createNSStringWithoutCopying().get()]);
    NSRange range = [attributedString doubleClickAtIndex:std::min<unsigned>(position, text.length() - 1)];
    *start = range.location;
    *end = range.location + range.length;
#else
    if (text.isEmpty()) {
        *start = 0;
        *end = 0;
        return;
    }

    if (static_cast<unsigned>(position) >= text.length()) {
        ASSERT_WITH_MESSAGE(static_cast<unsigned>(position) < text.length(), "position exceeds text.length()");
        *start = text.length() - 1;
        *end = text.length() - 1;
        return;
    }

    // For complex text (Thai, Japanese, Chinese), visible_units will pass the text in as a 
    // single contiguous run of characters, providing as much context as is possible.
    // We only need one character to determine if the text is complex.
    char32_t ch;
    unsigned i = position;
    U16_NEXT(text, i, text.length(), ch);
    bool isComplex = requiresContextForWordBoundary(ch);

    // FIXME: This check improves our word boundary behavior, but doesn't actually go far enough.
    // See <rdar://problem/8853951> Take complex word boundary finding path when necessary
    if (!isComplex) {
        // Check again for complex text, at the start of the run.
        i = 0;
        U16_NEXT(text, i, text.length(), ch);
        isComplex = requiresContextForWordBoundary(ch);
    }

    if (isComplex)
        findComplexWordBoundary(text, position, start, end);
    else
        findSimpleWordBoundary(text, position, start, end);

#define LOG_WORD_BREAK 0
#if LOG_WORD_BREAK
    auto uniString = text.createCFStringWithoutCopying();
    auto foundWord = text.substring(*start, *end - *start).createCFStringWithoutCopying();
    NSLog(@"%s_BREAK '%@' (%d,%d) in '%@' (%p) at %d, length=%d", isComplex ? "COMPLEX" : "SIMPLE", foundWord.get(), *start, *end, uniString.get(), uniString.get(), position, text.length());
#endif
    
#endif
}

void findEndWordBoundary(StringView text, int position, int* end)
{
    int start;
    findWordBoundary(text, position, &start, end);
}

#if USE(APPKIT)

// FIXME: Is this special Mac implementation actually important, or can
// we share with all the other platforms?
int findNextWordFromIndex(StringView text, int position, bool forward)
{   
    String textWithoutUnpairedSurrogates;
    if (hasUnpairedSurrogate(text)) {
        textWithoutUnpairedSurrogates = replaceUnpairedSurrogatesWithReplacementCharacter(text.toStringWithoutCopying());
        text = textWithoutUnpairedSurrogates;
    }
    auto attributedString = adoptNS([[NSAttributedString alloc] initWithString:text.createNSStringWithoutCopying().get()]);
    return [attributedString nextWordFromIndex:position forward:forward];
}

#endif // USE(APPKIT)

}
