/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 19, 2025.
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

#include "SourceSpan.h"
#include <wtf/text/WTFString.h>

namespace WGSL {

// https://www.w3.org/TR/WGSL/#keyword-summary
#define FOREACH_KEYWORD(F) \
    F(alias,        Alias) \
    F(break,        Break) \
    F(case,         Case) \
    F(const,        Const) \
    F(const_assert, ConstAssert) \
    F(continue,     Continue) \
    F(continuing,   Continuing) \
    F(default,      Default) \
    F(diagnostic,   Diagnostic) \
    F(discard,      Discard) \
    F(else,         Else) \
    F(enable,       Enable) \
    F(false,        False) \
    F(fn,           Fn) \
    F(for,          For) \
    F(if,           If) \
    F(let,          Let) \
    F(loop,         Loop) \
    F(override,     Override) \
    F(requires,     Requires) \
    F(return,       Return) \
    F(struct,       Struct) \
    F(switch,       Switch) \
    F(true,         True) \
    F(var,          Var) \
    F(while,        While)

enum class TokenType: uint32_t {
    // Instead of having this type, we could have a std::optional<Token> everywhere that we currently have a Token.
    // I made this choice for two reasons:
    // - space efficiency: we don't use an extra word of memory for the variant's tag
    //     (although this part could be solved by using https://github.com/akrzemi1/markable)
    // - ease of use and time efficiency: everywhere that we check for a given TokenType, we would have to first check that the Token is not nullopt, and then check the TokenType.
    Invalid,

    EndOfFile,

    AbstractFloatLiteral,
    IntegerLiteral,
    IntegerLiteralSigned,
    IntegerLiteralUnsigned,
    FloatLiteral,
    HalfLiteral,

    Identifier,

    ReservedWord,

#define ENUM_ENTRY(_, name) Keyword##name,
FOREACH_KEYWORD(ENUM_ENTRY)
#undef ENUM_ENTRY

    And,
    AndAnd,
    AndEq,
    Arrow,
    Attribute,
    Bang,
    BangEq,
    BraceLeft,
    BraceRight,
    BracketLeft,
    BracketRight,
    Colon,
    Comma,
    Equal,
    EqEq,
    Gt,
    GtEq,
    GtGt,
    GtGtEq,
    Lt,
    LtEq,
    LtLt,
    LtLtEq,
    Minus,
    MinusMinus,
    MinusEq,
    Modulo,
    ModuloEq,
    Or,
    OrOr,
    OrEq,
    ParenLeft,
    ParenRight,
    Period,
    Plus,
    PlusPlus,
    PlusEq,
    Semicolon,
    Slash,
    SlashEq,
    Star,
    StarEq,
    Tilde,
    Underbar,
    Xor,
    XorEq,

    Placeholder,
    TemplateArgsLeft,
    TemplateArgsRight,
};

String toString(TokenType);

struct Token {
    TokenType type;
    SourceSpan span;
    union {
        double floatValue;
        int64_t integerValue;
        String ident;
    };

    Token(TokenType type, SourcePosition position, unsigned length)
        : type(type)
        , span(position.line, position.lineOffset, position.offset, length)
    {
        ASSERT(type != TokenType::AbstractFloatLiteral
            && type != TokenType::Identifier
            && type != TokenType::IntegerLiteral
            && type != TokenType::IntegerLiteralSigned
            && type != TokenType::IntegerLiteralUnsigned
            && type != TokenType::FloatLiteral
            && type != TokenType::HalfLiteral);
    }

    Token(TokenType type, SourcePosition position, unsigned length, int64_t integerValue)
        : type(type)
        , span(position.line, position.lineOffset, position.offset, length)
        , integerValue(integerValue)
    {
        ASSERT(type == TokenType::IntegerLiteral
            || type == TokenType::IntegerLiteralSigned
            || type == TokenType::IntegerLiteralUnsigned);
    }

    Token(TokenType type, SourcePosition position, unsigned length, double floatValue)
        : type(type)
        , span(position.line, position.lineOffset, position.offset, length)
        , floatValue(floatValue)
    {
        ASSERT(type == TokenType::AbstractFloatLiteral
            || type == TokenType::FloatLiteral
            || type == TokenType::HalfLiteral);
    }

    Token(TokenType type, SourcePosition position, unsigned length, String&& ident)
        : type(type)
        , span(position.line, position.lineOffset, position.offset, length)
        , ident(WTFMove(ident))
    {
        ASSERT(this->ident.impl() && this->ident.impl()->bufferOwnership() == StringImpl::BufferInternal);
        ASSERT(type == TokenType::Identifier);
    }

    Token& operator=(Token&& other)
    {
        if (type == TokenType::Identifier)
            ident.~String();

        type = other.type;
        span = other.span;

        switch (other.type) {
        case TokenType::Identifier:
            new (NotNull, &ident) String();
            ident = other.ident;
            break;
        case TokenType::IntegerLiteral:
        case TokenType::IntegerLiteralSigned:
        case TokenType::IntegerLiteralUnsigned:
            integerValue = other.integerValue;
            break;
        case TokenType::AbstractFloatLiteral:
        case TokenType::FloatLiteral:
        case TokenType::HalfLiteral:
            floatValue = other.floatValue;
            break;
        default:
            break;
        }

        return *this;
    }

    Token& operator=(const Token& other)
    {
        type = other.type;
        span = other.span;

        switch (other.type) {
        case TokenType::Identifier:
            new (NotNull, &ident) String();
            ident = other.ident;
            break;
        case TokenType::IntegerLiteral:
        case TokenType::IntegerLiteralSigned:
        case TokenType::IntegerLiteralUnsigned:
            integerValue = other.integerValue;
            break;
        case TokenType::AbstractFloatLiteral:
        case TokenType::FloatLiteral:
        case TokenType::HalfLiteral:
            floatValue = other.floatValue;
            break;
        default:
            break;
        }

        return *this;
    }

    Token(const Token& other)
        : type(other.type)
        , span(other.span)
    {
        switch (other.type) {
        case TokenType::Identifier:
            new (NotNull, &ident) String();
            ident = other.ident;
            break;
        case TokenType::IntegerLiteral:
        case TokenType::IntegerLiteralSigned:
        case TokenType::IntegerLiteralUnsigned:
            integerValue = other.integerValue;
            break;
        case TokenType::AbstractFloatLiteral:
        case TokenType::FloatLiteral:
        case TokenType::HalfLiteral:
            floatValue = other.floatValue;
            break;
        default:
            break;
        }
    }

    ~Token()
    {
        if (type == TokenType::Identifier)
            (&ident)->~String();
    }
};

} // namespace WGSL
