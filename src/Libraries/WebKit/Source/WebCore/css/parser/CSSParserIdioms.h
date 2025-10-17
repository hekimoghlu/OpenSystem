/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 10, 2022.
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

#include "CSSParserContext.h"
#include "CSSValueKeywords.h"
#include <wtf/ASCIICType.h>

namespace WebCore {
    
// Space characters as defined by the CSS specification.
// http://www.w3.org/TR/css3-syntax/#whitespace

template<typename CharacterType>
inline bool isCSSSpace(CharacterType c)
{
    return c == ' ' || c == '\t' || c == '\n';
}

// http://dev.w3.org/csswg/css-syntax/#name-start-code-point
template <typename CharacterType>
bool isNameStartCodePoint(CharacterType c)
{
    return isASCIIAlpha(c) || c == '_' || !isASCII(c);
}

// http://dev.w3.org/csswg/css-syntax/#name-code-point
template <typename CharacterType>
bool isNameCodePoint(CharacterType c)
{
    return isNameStartCodePoint(c) || isASCIIDigit(c) || c == '-';
}

bool isColorKeywordAllowedInMode(CSSValueID, CSSParserMode);

inline bool isCSSWideKeyword(CSSValueID valueID)
{
    switch (valueID) {
    case CSSValueInitial:
    case CSSValueInherit:
    case CSSValueUnset:
    case CSSValueRevert:
    case CSSValueRevertLayer:
        return true;
    default:
        return false;
    };
}

inline bool isValidCustomIdentifier(CSSValueID valueID)
{
    // "default" is obsolete as a CSS-wide keyword but is still not allowed as a custom identifier.
    return !isCSSWideKeyword(valueID) && valueID != CSSValueDefault;
}

} // namespace WebCore
