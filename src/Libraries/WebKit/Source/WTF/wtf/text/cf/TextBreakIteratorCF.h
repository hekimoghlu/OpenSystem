/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 19, 2025.
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

#include <wtf/text/cf/TextBreakIteratorCFCharacterCluster.h>
#include <wtf/text/cf/TextBreakIteratorCFStringTokenizer.h>

namespace WTF {

class TextBreakIteratorCF {
    WTF_MAKE_FAST_ALLOCATED;
public:
    enum class Mode {
        ComposedCharacter,
        BackwardDeletion,
        Word,
        Sentence,
        Paragraph,
        LineBreak,
        WordBoundary,
    };

    TextBreakIteratorCF(StringView string, StringView priorContext, Mode mode, const AtomString& locale)
        : m_backing(mapModeToBackingIterator(string, priorContext, mode, locale))
    {
    }

    TextBreakIteratorCF() = delete;
    TextBreakIteratorCF(const TextBreakIteratorCF&) = delete;
    TextBreakIteratorCF(TextBreakIteratorCF&&) = default;
    TextBreakIteratorCF& operator=(const TextBreakIteratorCF&) = delete;
    TextBreakIteratorCF& operator=(TextBreakIteratorCF&&) = default;

    void setText(StringView string, std::span<const UChar> priorContext)
    {
        return switchOn(m_backing, [&](auto& iterator) {
            return iterator.setText(string, priorContext);
        });
    }

    std::optional<unsigned> preceding(unsigned location) const
    {
        return switchOn(m_backing, [&](const auto& iterator) {
            return iterator.preceding(location);
        });
    }

    std::optional<unsigned> following(unsigned location) const
    {
        return switchOn(m_backing, [&](const auto& iterator) {
            return iterator.following(location);
        });
    }

    bool isBoundary(unsigned location) const
    {
        return switchOn(m_backing, [&](const auto& iterator) {
            return iterator.isBoundary(location);
        });
    }

private:
    using BackingVariant = std::variant<TextBreakIteratorCFCharacterCluster, TextBreakIteratorCFStringTokenizer>;

    static BackingVariant mapModeToBackingIterator(StringView string, StringView priorContext, Mode mode, const AtomString& locale)
    {
        switch (mode) {
        case Mode::ComposedCharacter:
            return TextBreakIteratorCFCharacterCluster(string, priorContext, TextBreakIteratorCFCharacterCluster::Mode::ComposedCharacter);
        case Mode::BackwardDeletion:
            return TextBreakIteratorCFCharacterCluster(string, priorContext, TextBreakIteratorCFCharacterCluster::Mode::BackwardDeletion);
        case Mode::Word:
            return TextBreakIteratorCFStringTokenizer(string, priorContext, TextBreakIteratorCFStringTokenizer::Mode::Word, locale);
        case Mode::Sentence:
            return TextBreakIteratorCFStringTokenizer(string, priorContext, TextBreakIteratorCFStringTokenizer::Mode::Sentence, locale);
        case Mode::Paragraph:
            return TextBreakIteratorCFStringTokenizer(string, priorContext, TextBreakIteratorCFStringTokenizer::Mode::Paragraph, locale);
        case Mode::LineBreak:
            return TextBreakIteratorCFStringTokenizer(string, priorContext, TextBreakIteratorCFStringTokenizer::Mode::LineBreak, locale);
        case Mode::WordBoundary:
            return TextBreakIteratorCFStringTokenizer(string, priorContext, TextBreakIteratorCFStringTokenizer::Mode::WordBoundary, locale);
        }
    }

    BackingVariant m_backing;
};

} // namespace WTF
