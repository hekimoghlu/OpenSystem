/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 8, 2025.
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
#include <wtf/text/TextBreakIterator.h>

#include <wtf/Compiler.h>
#include <wtf/text/TextBreakIteratorInternalICU.h>
#include <wtf/text/icu/UTextProviderLatin1.h>
#include <wtf/text/icu/UTextProviderUTF16.h>
#include <atomic>
#include <unicode/ubrk.h>

namespace WTF {

TextBreakIteratorCache& TextBreakIteratorCache::singleton()
{
    static LazyNeverDestroyed<TextBreakIteratorCache> cache;
    static std::once_flag onceKey;
    std::call_once(onceKey, [&] {
        cache.construct();
    });
    return cache.get();
}

#if !PLATFORM(COCOA)

TextBreakIterator::Backing TextBreakIterator::mapModeToBackingIterator(StringView string, std::span<const UChar> priorContext, Mode mode, ContentAnalysis, const AtomString& locale)
{
    return switchOn(mode, [string, priorContext, &locale](TextBreakIterator::LineMode lineMode) -> TextBreakIterator::Backing {
        return TextBreakIteratorICU(string, priorContext, TextBreakIteratorICU::LineMode { lineMode.behavior }, locale);
    }, [string, priorContext, &locale](TextBreakIterator::CaretMode) -> TextBreakIterator::Backing {
        return TextBreakIteratorICU(string, priorContext, TextBreakIteratorICU::CharacterMode { }, locale);
    }, [string, priorContext, &locale](TextBreakIterator::DeleteMode) -> TextBreakIterator::Backing {
        return TextBreakIteratorICU(string, priorContext, TextBreakIteratorICU::CharacterMode { }, locale);
    }, [string, priorContext, &locale](TextBreakIterator::CharacterMode) -> TextBreakIterator::Backing {
        return TextBreakIteratorICU(string, priorContext, TextBreakIteratorICU::CharacterMode { }, locale);
    });
}

TextBreakIterator::TextBreakIterator(StringView string, std::span<const UChar> priorContext, Mode mode, ContentAnalysis contentAnalysis, const AtomString& locale)
    : m_backing(mapModeToBackingIterator(string, priorContext, mode, contentAnalysis, locale))
    , m_mode(mode)
    , m_locale(locale)
{
}

#endif

// Iterator initialization

static UBreakIterator* initializeIterator(UBreakIteratorType type, const char* locale = currentTextBreakLocaleID())
{
    UErrorCode openStatus = U_ZERO_ERROR;
    UBreakIterator* iterator = ubrk_open(type, locale, nullptr, 0, &openStatus);
    ASSERT_WITH_MESSAGE(U_SUCCESS(openStatus), "ICU could not open a break iterator: %s (%d)", u_errorName(openStatus), openStatus);
    return iterator;
}

// Iterator text setting

static UBreakIterator* setTextForIterator(UBreakIterator& iterator, StringView string)
{
    if (string.is8Bit()) {
        UTextWithBuffer textLocal;
        textLocal.text = UTEXT_INITIALIZER;
        textLocal.text.extraSize = sizeof(textLocal.buffer);
        textLocal.text.pExtra = textLocal.buffer;

        UErrorCode openStatus = U_ZERO_ERROR;
        UText* text = openLatin1UTextProvider(&textLocal, string.span8(), &openStatus);
        if (U_FAILURE(openStatus)) {
            LOG_ERROR("uTextOpenLatin1 failed with status %d", openStatus);
            return nullptr;
        }

        UErrorCode setTextStatus = U_ZERO_ERROR;
        ubrk_setUText(&iterator, text, &setTextStatus);
        if (U_FAILURE(setTextStatus)) {
            LOG_ERROR("ubrk_setUText failed with status %d", setTextStatus);
            return nullptr;
        }

        utext_close(text);
    } else {
        UErrorCode setTextStatus = U_ZERO_ERROR;
        auto characters = string.span16();
        ubrk_setText(&iterator, characters.data(), characters.size(), &setTextStatus);
        if (U_FAILURE(setTextStatus))
            return nullptr;
    }

    return &iterator;
}

// Static iterators

// FIXME: Delete this in favor of CachedTextBreakIterator.
UBreakIterator* wordBreakIterator(StringView string)
{
    static UBreakIterator* staticWordBreakIterator = initializeIterator(UBRK_WORD);
    if (!staticWordBreakIterator)
        return nullptr;

    return setTextForIterator(*staticWordBreakIterator, string);
}

// FIXME: Delete this in favor of CachedTextBreakIterator.
UBreakIterator* sentenceBreakIterator(StringView string)
{
    static UBreakIterator* staticSentenceBreakIterator = initializeIterator(UBRK_SENTENCE);
    if (!staticSentenceBreakIterator)
        return nullptr;

    return setTextForIterator(*staticSentenceBreakIterator, string);
}

ALLOW_DEPRECATED_PRAGMA_BEGIN
static std::atomic<UBreakIterator*> nonSharedCharacterBreakIterator = ATOMIC_VAR_INIT(nullptr);
ALLOW_DEPRECATED_PRAGMA_END

static inline UBreakIterator* getNonSharedCharacterBreakIterator()
{
    if (auto *res = nonSharedCharacterBreakIterator.exchange(nullptr, std::memory_order_acquire))
        return res;
    return initializeIterator(UBRK_CHARACTER);
}

static inline void cacheNonSharedCharacterBreakIterator(UBreakIterator* cacheMe)
{
    if (auto *old = nonSharedCharacterBreakIterator.exchange(cacheMe, std::memory_order_release))
        ubrk_close(old);
}

NonSharedCharacterBreakIterator::NonSharedCharacterBreakIterator(StringView string)
{
    if ((m_iterator = getNonSharedCharacterBreakIterator()))
        m_iterator = setTextForIterator(*m_iterator, string);
}

NonSharedCharacterBreakIterator::~NonSharedCharacterBreakIterator()
{
    if (m_iterator)
        cacheNonSharedCharacterBreakIterator(m_iterator);
}

NonSharedCharacterBreakIterator::NonSharedCharacterBreakIterator(NonSharedCharacterBreakIterator&& other)
    : m_iterator(nullptr)
{
    std::swap(m_iterator, other.m_iterator);
}

// Iterator implemenation.

bool isWordTextBreak(UBreakIterator* iterator)
{
    int ruleStatus = ubrk_getRuleStatus(iterator);
    return ruleStatus != UBRK_WORD_NONE;
}

unsigned numGraphemeClusters(StringView string)
{
    unsigned stringLength = string.length();
    
    if (!stringLength)
        return 0;

    // The only Latin-1 Extended Grapheme Cluster is CRLF.
    if (string.is8Bit()) {
        auto characters = string.span8();
        unsigned numCRLF = 0;
        for (unsigned i = 1; i < stringLength; ++i)
            numCRLF += characters[i - 1] == '\r' && characters[i] == '\n';
        return stringLength - numCRLF;
    }

    NonSharedCharacterBreakIterator iterator { string };
    if (!iterator) {
        ASSERT_NOT_REACHED();
        return stringLength;
    }

    unsigned numGraphemeClusters = 0;
    while (ubrk_next(iterator) != UBRK_DONE)
        ++numGraphemeClusters;
    return numGraphemeClusters;
}

unsigned numCodeUnitsInGraphemeClusters(StringView string, unsigned numGraphemeClusters)
{
    unsigned stringLength = string.length();

    if (stringLength <= numGraphemeClusters)
        return stringLength;

    // The only Latin-1 Extended Grapheme Cluster is CRLF.
    if (string.is8Bit()) {
        auto characters = string.span8();
        unsigned i, j;
        for (i = 0, j = 0; i < numGraphemeClusters && j + 1 < stringLength; ++i, ++j)
            j += characters[j] == '\r' && characters[j + 1] == '\n';
        return j + (i < numGraphemeClusters);
    }

    NonSharedCharacterBreakIterator iterator { string };
    if (!iterator) {
        ASSERT_NOT_REACHED();
        return stringLength;
    }

    for (unsigned i = 0; i < numGraphemeClusters; ++i) {
        if (ubrk_next(iterator) == UBRK_DONE)
            return stringLength;
    }
    return ubrk_current(iterator);
}

} // namespace WTF
