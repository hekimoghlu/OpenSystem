/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 17, 2022.
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
#include "URLFilterParser.h"

#if ENABLE(CONTENT_EXTENSIONS)

#include "CombinedURLFilters.h"
#include "Term.h"
#include <JavaScriptCore/YarrParser.h>
#include <wtf/Deque.h>
#include <wtf/text/CString.h>

namespace WebCore {

namespace ContentExtensions {

class PatternParser {
public:
    explicit PatternParser(bool patternIsCaseSensitive)
        : m_patternIsCaseSensitive(patternIsCaseSensitive)
        , m_parseStatus(URLFilterParser::Ok)
    {
    }

    void finalize(uint64_t patternId, CombinedURLFilters& combinedURLFilters)
    {
        if (hasError())
            return;

        sinkFloatingTermIfNecessary();

        simplifySunkTerms();

        // Check to see if there are any terms without ? or *.
        bool matchesEverything = true;
        for (const auto& term : m_sunkTerms) {
            if (term.matchesAtLeastOneCharacter()) {
                matchesEverything = false;
                break;
            }
        }
        if (matchesEverything) {
            fail(URLFilterParser::MatchesEverything);
            return;
        }

        combinedURLFilters.addPattern(patternId, m_sunkTerms);
    }

    URLFilterParser::ParseStatus parseStatus() const
    {
        return m_parseStatus;
    }

    void atomPatternCharacter(UChar character)
    {
        if (hasError())
            return;

        if (!isASCII(character)) {
            fail(URLFilterParser::NonASCII);
            return;
        }

        sinkFloatingTermIfNecessary();
        ASSERT(!m_floatingTerm.isValid());

        char asciiChararacter = static_cast<char>(character);
        m_floatingTerm = Term(asciiChararacter, m_patternIsCaseSensitive);
    }

    void atomBuiltInCharacterClass(JSC::Yarr::BuiltInCharacterClassID builtInCharacterClassID, bool inverted)
    {
        if (hasError())
            return;

        sinkFloatingTermIfNecessary();
        ASSERT(!m_floatingTerm.isValid());

        if (builtInCharacterClassID == JSC::Yarr::BuiltInCharacterClassID::DotClassID && !inverted)
            m_floatingTerm = Term(Term::UniversalTransition);
        else
            fail(URLFilterParser::UnsupportedCharacterClass);
    }

    void quantifyAtom(unsigned minimum, unsigned maximum, bool)
    {
        if (hasError())
            return;

        ASSERT(m_floatingTerm.isValid());

        if (!minimum && maximum == 1)
            m_floatingTerm.quantify(AtomQuantifier::ZeroOrOne);
        else if (!minimum && maximum == JSC::Yarr::quantifyInfinite)
            m_floatingTerm.quantify(AtomQuantifier::ZeroOrMore);
        else if (minimum == 1 && maximum == JSC::Yarr::quantifyInfinite)
            m_floatingTerm.quantify(AtomQuantifier::OneOrMore);
        else
            fail(URLFilterParser::InvalidQuantifier);
    }

    void atomBackReference(unsigned)
    {
        fail(URLFilterParser::BackReference);
    }

    void atomNamedBackReference(const String&)
    {
        fail(URLFilterParser::BackReference);
    }

    void atomNamedForwardReference(const String&)
    {
        fail(URLFilterParser::ForwardReference);
    }
    
    void assertionBOL()
    {
        if (hasError())
            return;

        if (m_floatingTerm.isValid() || !m_sunkTerms.isEmpty() || !m_openGroups.isEmpty()) {
            fail(URLFilterParser::MisplacedStartOfLine);
            return;
        }

        m_hasBeginningOfLineAssertion = true;
    }

    void assertionEOL()
    {
        if (hasError())
            return;

        sinkFloatingTermIfNecessary();
        ASSERT(!m_floatingTerm.isValid());

        m_floatingTerm = Term(Term::EndOfLineAssertionTerm);
    }

    void assertionWordBoundary(bool)
    {
        fail(URLFilterParser::WordBoundary);
    }

    void atomCharacterClassBegin(bool inverted = false)
    {
        if (hasError())
            return;

        sinkFloatingTermIfNecessary();
        ASSERT(!m_floatingTerm.isValid());

        m_floatingTerm = Term(Term::CharacterSetTerm, inverted);
    }

    void atomCharacterClassAtom(UChar character)
    {
        if (hasError())
            return;

        ASSERT(isASCII(character));

        m_floatingTerm.addCharacter(character, m_patternIsCaseSensitive);
    }

    void atomCharacterClassRange(UChar a, UChar b)
    {
        if (hasError())
            return;

        ASSERT(a);
        ASSERT(b);
        ASSERT(isASCII(a));
        ASSERT(isASCII(b));

        for (unsigned i = a; i <= b; ++i)
            m_floatingTerm.addCharacter(static_cast<UChar>(i), m_patternIsCaseSensitive);
    }

    void atomClassStringDisjunction(Vector<Vector<char32_t>>)
    {
        fail(URLFilterParser::AtomCharacter);
    }

    void atomCharacterClassSetOp(JSC::Yarr::CharacterClassSetOp)
    {
        // Nothing to do here.
    }

    void atomCharacterClassPushNested()
    {
        // Nothing to do here.
    }

    void atomCharacterClassPopNested()
    {
        // Nothing to do here.
    }

    void atomCharacterClassEnd()
    {
        // Nothing to do here. The character set atom may have a quantifier, we sink the atom lazily.
    }

    void atomCharacterClassBuiltIn(JSC::Yarr::BuiltInCharacterClassID, bool)
    {
        fail(URLFilterParser::AtomCharacter);
    }

    void atomParenthesesSubpatternBegin(bool = true, std::optional<String> = std::nullopt)
    {
        if (hasError())
            return;

        sinkFloatingTermIfNecessary();

        m_openGroups.append(Term(Term::GroupTerm));
    }

    void atomParentheticalAssertionBegin(bool, MatchDirection)
    {
        fail(URLFilterParser::Group);
    }

    void atomParenthesesEnd()
    {
        if (hasError())
            return;

        sinkFloatingTermIfNecessary();
        ASSERT(!m_floatingTerm.isValid());

        m_floatingTerm = m_openGroups.takeLast();
    }

    void disjunction(JSC::Yarr::CreateDisjunctionPurpose)
    {
        fail(URLFilterParser::Disjunction);
    }

    NO_RETURN_DUE_TO_CRASH void resetForReparsing()
    {
        RELEASE_ASSERT_NOT_REACHED();
    }

    constexpr static bool abortedDueToError() { return false; }
    constexpr static JSC::Yarr::ErrorCode abortErrorCode() { return JSC::Yarr::ErrorCode::NoError; }

private:
    bool hasError() const
    {
        return m_parseStatus != URLFilterParser::Ok;
    }

    void fail(URLFilterParser::ParseStatus reason)
    {
        if (hasError())
            return;

        m_parseStatus = reason;
    }

    void sinkFloatingTermIfNecessary()
    {
        if (!m_floatingTerm.isValid())
            return;

        if (m_hasProcessedEndOfLineAssertion) {
            fail(URLFilterParser::MisplacedEndOfLine);
            m_floatingTerm = Term();
            return;
        }

        if (m_floatingTerm.isEndOfLineAssertion())
            m_hasProcessedEndOfLineAssertion = true;

        if (!m_openGroups.isEmpty()) {
            m_openGroups.last().extendGroupSubpattern(m_floatingTerm);
            m_floatingTerm = Term();
            return;
        }

        m_sunkTerms.append(m_floatingTerm);
        m_floatingTerm = Term();
    }

    void simplifySunkTerms()
    {
        ASSERT(!m_floatingTerm.isValid());

        if (m_sunkTerms.isEmpty())
            return;

        Term canonicalDotStar(Term::UniversalTransition);
        canonicalDotStar.quantify(AtomQuantifier::ZeroOrMore);

        // Replace every ".*"-like terms by our canonical version. Remove any duplicate ".*".
        {
            unsigned termIndex = 0;
            bool isAfterDotStar = false;
            while (termIndex < m_sunkTerms.size()) {
                if (isAfterDotStar && m_sunkTerms[termIndex].isKnownToMatchAnyString()) {
                    m_sunkTerms.remove(termIndex);
                    continue;
                }
                isAfterDotStar = false;

                if (m_sunkTerms[termIndex].isKnownToMatchAnyString()) {
                    m_sunkTerms[termIndex] = canonicalDotStar;
                    isAfterDotStar = true;
                }
                ++termIndex;
            }
        }

        // Add our ".*" in front if needed.
        if (!m_hasBeginningOfLineAssertion && !m_sunkTerms.first().isKnownToMatchAnyString())
            m_sunkTerms.insert(0, canonicalDotStar);

        // Remove trailing ".*$".
        if (m_sunkTerms.size() > 2 && m_sunkTerms.last().isEndOfLineAssertion() && m_sunkTerms[m_sunkTerms.size() - 2].isKnownToMatchAnyString())
            m_sunkTerms.shrink(m_sunkTerms.size() - 2);

        // Remove irrelevant terms that can match empty. For example in "foob?", matching "b" is irrelevant.
        if (m_sunkTerms.last().isEndOfLineAssertion())
            return;
        while (!m_sunkTerms.isEmpty() && !m_sunkTerms.last().matchesAtLeastOneCharacter())
            m_sunkTerms.removeLast();
    }

    bool m_patternIsCaseSensitive;

    Deque<Term> m_openGroups;
    Vector<Term> m_sunkTerms;
    Term m_floatingTerm;
    bool m_hasBeginningOfLineAssertion { false };
    bool m_hasProcessedEndOfLineAssertion { false };

    URLFilterParser::ParseStatus m_parseStatus;
};

URLFilterParser::URLFilterParser(CombinedURLFilters& combinedURLFilters)
    : m_combinedURLFilters(combinedURLFilters)
{
}

URLFilterParser::~URLFilterParser() = default;

URLFilterParser::ParseStatus URLFilterParser::addPattern(StringView pattern, bool patternIsCaseSensitive, uint64_t patternId)
{
    if (!pattern.containsOnlyASCII())
        return NonASCII;
    if (pattern.isEmpty())
        return EmptyPattern;

    ParseStatus status = Ok;
    PatternParser patternParser(patternIsCaseSensitive);
    if (!JSC::Yarr::hasError(JSC::Yarr::parse(patternParser, pattern, JSC::Yarr::CompileMode::Legacy, 0, false)))
        patternParser.finalize(patternId, m_combinedURLFilters);
    else
        status = YarrError;
    
    if (status == Ok)
        status = patternParser.parseStatus();

    return status;
}

ASCIILiteral URLFilterParser::statusString(ParseStatus status)
{
    switch (status) {
    case Ok:
        return "Ok"_s;
    case MatchesEverything:
        return "Matches everything."_s;
    case NonASCII:
        return "Only ASCII characters are supported in pattern."_s;
    case UnsupportedCharacterClass:
        return "Character class is not supported."_s;
    case BackReference:
        return "Patterns cannot contain backreferences."_s;
    case ForwardReference:
        return "Patterns cannot contain forward references."_s;
    case MisplacedStartOfLine:
        return "Start of line assertion can only appear as the first term in a filter."_s;
    case WordBoundary:
        return "Word boundaries assertions are not supported yet."_s;
    case AtomCharacter:
        return "Builtins character class atoms are not supported yet."_s;
    case Group:
        return "Groups are not supported yet."_s;
    case Disjunction:
        return "Disjunctions are not supported yet."_s;
    case MisplacedEndOfLine:
        return "The end of line assertion must be the last term in an expression."_s;
    case EmptyPattern:
        return "Empty pattern."_s;
    case YarrError:
        return "Internal error in YARR."_s;
    case InvalidQuantifier:
        return "Arbitrary atom repetitions are not supported."_s;
    }

    RELEASE_ASSERT_NOT_REACHED();
}
    
} // namespace ContentExtensions
} // namespace WebCore

#endif // ENABLE(CONTENT_EXTENSIONS)
