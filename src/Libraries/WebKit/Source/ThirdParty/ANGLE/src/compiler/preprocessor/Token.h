/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 14, 2022.
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

#ifndef COMPILER_PREPROCESSOR_TOKEN_H_
#define COMPILER_PREPROCESSOR_TOKEN_H_

#include <ostream>
#include <string>

#include "compiler/preprocessor/SourceLocation.h"

namespace angle
{

namespace pp
{

struct Token
{
    enum Type
    {
        // Calling this ERROR causes a conflict with wingdi.h
        GOT_ERROR = -1,
        LAST      = 0,  // EOF.

        IDENTIFIER = 258,

        CONST_INT,
        CONST_FLOAT,

        OP_INC,
        OP_DEC,
        OP_LEFT,
        OP_RIGHT,
        OP_LE,
        OP_GE,
        OP_EQ,
        OP_NE,
        OP_AND,
        OP_XOR,
        OP_OR,
        OP_ADD_ASSIGN,
        OP_SUB_ASSIGN,
        OP_MUL_ASSIGN,
        OP_DIV_ASSIGN,
        OP_MOD_ASSIGN,
        OP_LEFT_ASSIGN,
        OP_RIGHT_ASSIGN,
        OP_AND_ASSIGN,
        OP_XOR_ASSIGN,
        OP_OR_ASSIGN,

        // Preprocessing token types.
        // These types are used by the preprocessor internally.
        // Preprocessor clients must not depend or check for them.
        PP_HASH,
        PP_NUMBER,
        PP_OTHER
    };
    enum Flags
    {
        AT_START_OF_LINE   = 1 << 0,
        HAS_LEADING_SPACE  = 1 << 1,
        EXPANSION_DISABLED = 1 << 2
    };

    Token() : type(0), flags(0) {}

    void reset();
    bool equals(const Token &other) const;

    // Returns true if this is the first token on line.
    // It disregards any leading whitespace.
    bool atStartOfLine() const { return (flags & AT_START_OF_LINE) != 0; }
    void setAtStartOfLine(bool start);

    bool hasLeadingSpace() const { return (flags & HAS_LEADING_SPACE) != 0; }
    void setHasLeadingSpace(bool space);

    bool expansionDisabled() const { return (flags & EXPANSION_DISABLED) != 0; }
    void setExpansionDisabled(bool disable);

    // Converts text into numeric value for CONST_INT and CONST_FLOAT token.
    // Returns false if the parsed value cannot fit into an int or float.
    bool iValue(int *value) const;
    bool uValue(unsigned int *value) const;

    int type;
    unsigned int flags;
    SourceLocation location;
    std::string text;
};

inline bool operator==(const Token &lhs, const Token &rhs)
{
    return lhs.equals(rhs);
}

inline bool operator!=(const Token &lhs, const Token &rhs)
{
    return !lhs.equals(rhs);
}

std::ostream &operator<<(std::ostream &out, const Token &token);

constexpr char kDefined[] = "defined";

}  // namespace pp

}  // namespace angle

#endif  // COMPILER_PREPROCESSOR_TOKEN_H_
