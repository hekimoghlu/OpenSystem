/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 5, 2022.
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

#include <CoreFoundation/CoreFoundation.h>
#include <wtf/RetainPtr.h>
#include <wtf/spi/cf/CFStringSPI.h>
#include <wtf/text/AtomString.h>
#include <wtf/text/StringView.h>
#include <wtf/text/cocoa/ContextualizedCFString.h>

namespace WTF {

class TextBreakIteratorCFStringTokenizer {
    WTF_MAKE_FAST_ALLOCATED;
public:
    enum class Mode {
        Word,
        Sentence,
        Paragraph,
        LineBreak,
        WordBoundary,
    };

    TextBreakIteratorCFStringTokenizer(StringView string, StringView priorContext, Mode mode, const AtomString& locale)
    {
        auto options = [mode] {
            switch (mode) {
            case Mode::Word:
                return kCFStringTokenizerUnitWord;
            case Mode::Sentence:
                return kCFStringTokenizerUnitSentence;
            case Mode::Paragraph:
                return kCFStringTokenizerUnitParagraph;
            case Mode::LineBreak:
                return kCFStringTokenizerUnitLineBreak;
            case Mode::WordBoundary:
                return kCFStringTokenizerUnitWordBoundary;
            }
        }();

        auto stringObject = createString(string, priorContext);
        m_stringLength = string.length();
        m_priorContextLength = priorContext.length();
        auto localeObject = adoptCF(CFLocaleCreate(kCFAllocatorDefault, locale.string().createCFString().get()));
        m_stringTokenizer = adoptCF(CFStringTokenizerCreate(kCFAllocatorDefault, stringObject.get(), CFRangeMake(0, m_stringLength + m_priorContextLength), options, localeObject.get()));
        if (!m_stringTokenizer)
            m_stringTokenizer = adoptCF(CFStringTokenizerCreate(kCFAllocatorDefault, stringObject.get(), CFRangeMake(0, m_stringLength + m_priorContextLength), options, nullptr));
        ASSERT(m_stringTokenizer);
    }

    TextBreakIteratorCFStringTokenizer() = delete;
    TextBreakIteratorCFStringTokenizer(const TextBreakIteratorCFStringTokenizer&) = delete;
    TextBreakIteratorCFStringTokenizer(TextBreakIteratorCFStringTokenizer&&) = default;
    TextBreakIteratorCFStringTokenizer& operator=(const TextBreakIteratorCFStringTokenizer&) = delete;
    TextBreakIteratorCFStringTokenizer& operator=(TextBreakIteratorCFStringTokenizer&&) = default;

    void setText(StringView string, StringView priorContext)
    {
        auto stringObject = createString(string, priorContext);
        m_stringLength = string.length();
        m_priorContextLength = priorContext.length();
        CFStringTokenizerSetString(m_stringTokenizer.get(), stringObject.get(), CFRangeMake(0, m_stringLength));
    }

    std::optional<unsigned> preceding(unsigned location) const
    {
        if (!location)
            return { };
        if (location > m_stringLength)
            return m_stringLength;
        CFStringTokenizerGoToTokenAtIndex(m_stringTokenizer.get(), location - 1 + m_priorContextLength);
        auto range = CFStringTokenizerGetCurrentTokenRange(m_stringTokenizer.get());
        if (range.location == kCFNotFound)
            return { };
        return std::max(static_cast<unsigned long>(range.location), m_priorContextLength) - m_priorContextLength;
    }

    std::optional<unsigned> following(unsigned location) const
    {
        if (location >= m_stringLength)
            return { };
        CFStringTokenizerGoToTokenAtIndex(m_stringTokenizer.get(), location + m_priorContextLength);
        auto range = CFStringTokenizerGetCurrentTokenRange(m_stringTokenizer.get());
        if (range.location == kCFNotFound)
            return { };
        return range.location + range.length - m_priorContextLength;
    }

    bool isBoundary(unsigned location) const
    {
        if (location == m_stringLength)
            return true;
        CFStringTokenizerGoToTokenAtIndex(m_stringTokenizer.get(), location + m_priorContextLength);
        auto range = CFStringTokenizerGetCurrentTokenRange(m_stringTokenizer.get());
        if (range.location == kCFNotFound)
            return true;
        return static_cast<unsigned long>(range.location) == location + m_priorContextLength;
    }

private:
    RetainPtr<CFStringRef> createString(StringView string, StringView priorContext)
    {
        if (priorContext.isEmpty())
            return string.createCFStringWithoutCopying();
        return createContextualizedCFString(string, priorContext);
    }

    RetainPtr<CFStringTokenizerRef> m_stringTokenizer;
    unsigned long m_stringLength { 0 };
    unsigned long m_priorContextLength { 0 };
};

}
