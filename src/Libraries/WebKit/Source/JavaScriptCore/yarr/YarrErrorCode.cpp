/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 19, 2024.
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
#include "YarrErrorCode.h"

#include "Error.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC { namespace Yarr {

ASCIILiteral errorMessage(ErrorCode error)
{
#define REGEXP_ERROR_PREFIX "Invalid regular expression: "
    // The order of this array must match the ErrorCode enum.
    static const ASCIILiteral errorMessages[] = {
        { },                                                                          // NoError

        // The following are hard errors.
        REGEXP_ERROR_PREFIX "regular expression too large"_s,                         // PatternTooLarge
        REGEXP_ERROR_PREFIX "numbers out of order in {} quantifier"_s,                // QuantifierOutOfOrder
        REGEXP_ERROR_PREFIX "nothing to repeat"_s,                                    // QuantifierWithoutAtom
        REGEXP_ERROR_PREFIX "number too large in {} quantifier"_s,                    // QuantifierTooLarge
        REGEXP_ERROR_PREFIX "incomplete {} quantifier for Unicode pattern"_s,         // QuantifierIncomplete
        REGEXP_ERROR_PREFIX "invalid quantifier"_s,                                   // CantQuantifyAtom
        REGEXP_ERROR_PREFIX "missing )"_s,                                            // MissingParentheses
        REGEXP_ERROR_PREFIX "unmatched ] or } bracket for Unicode pattern"_s,         // BracketUnmatched
        REGEXP_ERROR_PREFIX "unmatched parentheses"_s,                                // ParenthesesUnmatched
        REGEXP_ERROR_PREFIX "unrecognized character after (?"_s,                      // ParenthesesTypeInvalid
        REGEXP_ERROR_PREFIX "invalid group specifier name"_s,                         // InvalidGroupName
        REGEXP_ERROR_PREFIX "duplicate group specifier name"_s,                       // DuplicateGroupName
        REGEXP_ERROR_PREFIX "missing terminating ] for character class"_s,            // CharacterClassUnmatched
        REGEXP_ERROR_PREFIX "range out of order in character class"_s,                // CharacterClassRangeOutOfOrder
        REGEXP_ERROR_PREFIX "invalid range in character class for Unicode pattern"_s, // CharacterClassRangeInvalid
        REGEXP_ERROR_PREFIX "missing terminating } for class string disjunction"_s,   // ClassStringDisjunctionUnmatched
        REGEXP_ERROR_PREFIX "\\ at end of pattern"_s,                                 // EscapeUnterminated
        REGEXP_ERROR_PREFIX "invalid Unicode \\u escape"_s,                           // InvalidUnicodeEscape
        REGEXP_ERROR_PREFIX "invalid Unicode code point \\u{} escape"_s,              // InvalidUnicodeCodePointEscape
        REGEXP_ERROR_PREFIX "invalid backreference for Unicode pattern"_s,            // InvalidBackreference
        REGEXP_ERROR_PREFIX "invalid \\k<> named backreference"_s,                    // InvalidNamedBackReference
        REGEXP_ERROR_PREFIX "invalid escaped character for Unicode pattern"_s,        // InvalidIdentityEscape
        REGEXP_ERROR_PREFIX "invalid octal escape for Unicode pattern"_s,             // InvalidOctalEscape
        REGEXP_ERROR_PREFIX "invalid \\c escape for Unicode pattern"_s,               // InvalidControlLetterEscape
        REGEXP_ERROR_PREFIX "invalid property expression"_s,                          // InvalidUnicodePropertyExpression
        REGEXP_ERROR_PREFIX "pattern exceeds string length limits"_s,                 // OffsetTooLarge
        REGEXP_ERROR_PREFIX "invalid flags"_s,                                        // InvalidRegularExpressionFlags
        REGEXP_ERROR_PREFIX "invalid operation in class set"_s,                       // InvalidClassSetOperation
        REGEXP_ERROR_PREFIX "negated class set may contain strings"_s,                // NegatedClassSetMayContainStrings
        REGEXP_ERROR_PREFIX "invalid class set character"_s,                          // InvalidClassSetCharacter

        // The following are NOT hard errors.
        REGEXP_ERROR_PREFIX "too many nested disjunctions"_s,                         // TooManyDisjunctions
    };

    return errorMessages[static_cast<unsigned>(error)];
}

JSObject* errorToThrow(JSGlobalObject* globalObject, ErrorCode error)
{
    switch (error) {
    case ErrorCode::NoError:
        ASSERT_NOT_REACHED();
        return nullptr;
    case ErrorCode::PatternTooLarge:
    case ErrorCode::QuantifierOutOfOrder:
    case ErrorCode::QuantifierWithoutAtom:
    case ErrorCode::QuantifierTooLarge:
    case ErrorCode::QuantifierIncomplete:
    case ErrorCode::CantQuantifyAtom:
    case ErrorCode::MissingParentheses:
    case ErrorCode::BracketUnmatched:
    case ErrorCode::ParenthesesUnmatched:
    case ErrorCode::ParenthesesTypeInvalid:
    case ErrorCode::InvalidGroupName:
    case ErrorCode::DuplicateGroupName:
    case ErrorCode::CharacterClassUnmatched:
    case ErrorCode::CharacterClassRangeOutOfOrder:
    case ErrorCode::CharacterClassRangeInvalid:
    case ErrorCode::ClassStringDisjunctionUnmatched:
    case ErrorCode::EscapeUnterminated:
    case ErrorCode::InvalidUnicodeEscape:
    case ErrorCode::InvalidUnicodeCodePointEscape:
    case ErrorCode::InvalidBackreference:
    case ErrorCode::InvalidNamedBackReference:
    case ErrorCode::InvalidIdentityEscape:
    case ErrorCode::InvalidOctalEscape:
    case ErrorCode::InvalidControlLetterEscape:
    case ErrorCode::InvalidUnicodePropertyExpression:
    case ErrorCode::OffsetTooLarge:
    case ErrorCode::InvalidRegularExpressionFlags:
    case ErrorCode::InvalidClassSetOperation:
    case ErrorCode::NegatedClassSetMayContainStrings:
    case ErrorCode::InvalidClassSetCharacter:
        return createSyntaxError(globalObject, errorMessage(error));
    case ErrorCode::TooManyDisjunctions:
        return createOutOfMemoryError(globalObject, errorMessage(error));
    }

    ASSERT_NOT_REACHED();
    return nullptr;
}

} } // namespace JSC::Yarr

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
