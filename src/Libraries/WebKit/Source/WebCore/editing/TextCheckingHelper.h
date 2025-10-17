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

#include "ExceptionOr.h"
#include "SimpleRange.h"
#include "TextChecking.h"

namespace WebCore {

class EditorClient;
class LocalFrame;
class Position;
class TextCheckerClient;
class VisibleSelection;

struct TextCheckingResult;

// FIXME: Should move this class to its own header.
class TextCheckingParagraph {
public:
    explicit TextCheckingParagraph(const SimpleRange& checkingAndAutomaticReplacementRange);
    TextCheckingParagraph(const SimpleRange& checkingRange, const SimpleRange& automaticReplacementRange, const std::optional<SimpleRange>& paragraphRange);

    uint64_t rangeLength() const;
    SimpleRange subrange(CharacterRange) const;
    ExceptionOr<uint64_t> offsetTo(const Position&) const;
    void expandRangeToNextEnd();

    StringView text() const;

    bool isEmpty() const;

    uint64_t checkingStart() const;
    uint64_t checkingEnd() const;
    uint64_t checkingLength() const;
    StringView checkingSubstring() const { return text().substring(checkingStart(), checkingLength()); }

    uint64_t automaticReplacementStart() const;
    uint64_t automaticReplacementLength() const;

    bool checkingRangeMatches(CharacterRange range) const { return range.location == checkingStart() && range.length == checkingLength(); }
    bool isCheckingRangeCoveredBy(CharacterRange range) const { return range.location <= checkingStart() && range.location + range.length >= checkingStart() + checkingLength(); }
    bool checkingRangeCovers(CharacterRange range) const { return range.location < checkingEnd() && range.location + range.length > checkingStart(); }

    const SimpleRange& paragraphRange() const;

private:
    void invalidateParagraphRangeValues();
    const SimpleRange& offsetAsRange() const;

    SimpleRange m_checkingRange;
    SimpleRange m_automaticReplacementRange;
    mutable std::optional<SimpleRange> m_paragraphRange;
    mutable std::optional<SimpleRange> m_offsetAsRange;
    mutable String m_text;
    mutable std::optional<uint64_t> m_checkingStart;
    mutable std::optional<uint64_t> m_checkingLength;
    mutable std::optional<uint64_t> m_automaticReplacementStart;
    mutable std::optional<uint64_t> m_automaticReplacementLength;
};

class TextCheckingHelper {
public:
    TextCheckingHelper(EditorClient&, const SimpleRange&);

    struct MisspelledWord {
        String word;
        uint64_t offset { 0 };
    };
    struct UngrammaticalPhrase {
        String phrase;
        uint64_t offset { 0 };
        GrammarDetail detail;
    };

    MisspelledWord findFirstMisspelledWord() const;
    UngrammaticalPhrase findFirstUngrammaticalPhrase() const;
    std::variant<MisspelledWord, UngrammaticalPhrase> findFirstMisspelledWordOrUngrammaticalPhrase(bool checkGrammar) const;

    std::optional<SimpleRange> markAllMisspelledWords() const; // Returns the range of the first misspelled word.
    void markAllUngrammaticalPhrases() const;

    TextCheckingGuesses guessesForMisspelledWordOrUngrammaticalPhrase(bool checkGrammar) const;

private:
    enum class Operation : bool { FindFirst, MarkAll };
    std::pair<MisspelledWord, std::optional<SimpleRange>> findMisspelledWords(Operation) const; // Returns the first.
    UngrammaticalPhrase findUngrammaticalPhrases(Operation) const; // Returns the first.
    bool unifiedTextCheckerEnabled() const;
    int findUngrammaticalPhrases(Operation, const Vector<GrammarDetail>&, uint64_t badGrammarPhraseLocation, uint64_t startOffset, uint64_t endOffset) const;

    EditorClient& m_client;
    SimpleRange m_range;
};

void checkTextOfParagraph(TextCheckerClient&, StringView, OptionSet<TextCheckingType>, Vector<TextCheckingResult>&, const VisibleSelection& currentSelection);

bool unifiedTextCheckerEnabled(const LocalFrame*);
bool platformDrivenTextCheckerEnabled();
bool platformOrClientDrivenTextCheckerEnabled();

} // namespace WebCore
