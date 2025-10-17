/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 22, 2025.
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

#include <wtf/Forward.h>

namespace JSC {

class CallFrame;
class JSGlobalObject;
class JSObject;

namespace Yarr {

enum class ErrorCode : uint8_t {
    NoError = 0,

    // A hard error means that no matter what string the RegExp is evaluated on, it will
    // always fail. A SyntaxError is a hard error because the RegExp will never succeed no
    // matter what string it is run on. An OOME is not a hard error because the RegExp may
    // succeed when run on a different string.

    // The following are hard errors.
    PatternTooLarge,
    QuantifierOutOfOrder,
    QuantifierWithoutAtom,
    QuantifierTooLarge,
    QuantifierIncomplete,
    CantQuantifyAtom,
    MissingParentheses,
    BracketUnmatched,
    ParenthesesUnmatched,
    ParenthesesTypeInvalid,
    InvalidGroupName,
    DuplicateGroupName,
    CharacterClassUnmatched,
    CharacterClassRangeOutOfOrder,
    CharacterClassRangeInvalid,
    ClassStringDisjunctionUnmatched,
    EscapeUnterminated,
    InvalidUnicodeEscape,
    InvalidUnicodeCodePointEscape,
    InvalidBackreference,
    InvalidNamedBackReference,
    InvalidIdentityEscape,
    InvalidOctalEscape,
    InvalidControlLetterEscape,
    InvalidUnicodePropertyExpression,
    OffsetTooLarge,
    InvalidRegularExpressionFlags,
    InvalidClassSetOperation,
    NegatedClassSetMayContainStrings,
    InvalidClassSetCharacter,

    // The following are NOT hard errors.
    TooManyDisjunctions, // we ran out stack compiling.
};

JS_EXPORT_PRIVATE ASCIILiteral errorMessage(ErrorCode);
inline bool hasError(ErrorCode errorCode)
{
    return errorCode != ErrorCode::NoError;
}

inline bool hasHardError(ErrorCode errorCode)
{
    // See comment in the enum class ErrorCode above for the definition of hard errors.
    return hasError(errorCode) && errorCode < ErrorCode::TooManyDisjunctions;
}
JS_EXPORT_PRIVATE JSObject* errorToThrow(JSGlobalObject*, ErrorCode);

} } // namespace JSC::Yarr
