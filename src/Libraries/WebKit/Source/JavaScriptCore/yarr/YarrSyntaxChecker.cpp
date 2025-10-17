/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 11, 2022.
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
#include "YarrSyntaxChecker.h"

#include "YarrFlags.h"
#include "YarrParser.h"

namespace JSC { namespace Yarr {

class SyntaxChecker {
public:
    void assertionBOL() { }
    void assertionEOL() { }
    void assertionWordBoundary(bool) { }
    void atomPatternCharacter(char32_t) { }
    void atomBuiltInCharacterClass(BuiltInCharacterClassID, bool) { }
    void atomCharacterClassBegin(bool = false) { }
    void atomCharacterClassAtom(UChar) { }
    void atomCharacterClassRange(UChar, UChar) { }
    void atomCharacterClassBuiltIn(BuiltInCharacterClassID, bool) { }
    void atomClassStringDisjunction(Vector<Vector<char32_t>>&) { }
    void atomCharacterClassSetOp(CharacterClassSetOp) { }
    void atomCharacterClassPushNested() { }
    void atomCharacterClassPopNested() { }
    void atomCharacterClassEnd() { }
    void atomParenthesesSubpatternBegin(bool = true, std::optional<String> = std::nullopt) { }
    void atomParentheticalAssertionBegin(bool, MatchDirection) { }
    void atomParenthesesEnd() { }
    void atomBackReference(unsigned) { }
    void atomNamedBackReference(const String&) { }
    void atomNamedForwardReference(const String&) { }
    void quantifyAtom(unsigned, unsigned, bool) { }
    void disjunction(CreateDisjunctionPurpose) { }
    void resetForReparsing() { }

    constexpr static bool abortedDueToError() { return false; }
    constexpr static ErrorCode abortErrorCode() { return ErrorCode::NoError; }
};

ErrorCode checkSyntax(StringView pattern, StringView flags)
{
    SyntaxChecker syntaxChecker;

    auto parsedFlags = parseFlags(flags);
    if (!parsedFlags)
        return ErrorCode::InvalidRegularExpressionFlags;

    return parse(syntaxChecker, pattern, compileMode(parsedFlags));
}

}} // JSC::Yarr
