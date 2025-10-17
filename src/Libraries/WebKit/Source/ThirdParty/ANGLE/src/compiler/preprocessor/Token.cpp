/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 24, 2022.
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

//
// Copyright 2011 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/preprocessor/Token.h"

#include "common/debug.h"
#include "compiler/preprocessor/numeric_lex.h"

namespace angle
{

namespace pp
{

void Token::reset()
{
    type     = 0;
    flags    = 0;
    location = SourceLocation();
    text.clear();
}

bool Token::equals(const Token &other) const
{
    return (type == other.type) && (flags == other.flags) && (location == other.location) &&
           (text == other.text);
}

void Token::setAtStartOfLine(bool start)
{
    if (start)
        flags |= AT_START_OF_LINE;
    else
        flags &= ~AT_START_OF_LINE;
}

void Token::setHasLeadingSpace(bool space)
{
    if (space)
        flags |= HAS_LEADING_SPACE;
    else
        flags &= ~HAS_LEADING_SPACE;
}

void Token::setExpansionDisabled(bool disable)
{
    if (disable)
        flags |= EXPANSION_DISABLED;
    else
        flags &= ~EXPANSION_DISABLED;
}

bool Token::iValue(int *value) const
{
    ASSERT(type == CONST_INT);
    return numeric_lex_int(text, value);
}

bool Token::uValue(unsigned int *value) const
{
    ASSERT(type == CONST_INT);
    return numeric_lex_int(text, value);
}

std::ostream &operator<<(std::ostream &out, const Token &token)
{
    if (token.hasLeadingSpace())
        out << " ";

    out << token.text;
    return out;
}

}  // namespace pp

}  // namespace angle
