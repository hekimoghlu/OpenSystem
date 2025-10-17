/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 17, 2023.
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
#include "Token.h"

namespace WGSL {

String toString(TokenType type)
{
    switch (type) {
    case TokenType::Invalid:
        return "Invalid"_s;
    case TokenType::EndOfFile:
        return "EOF"_s;
    case TokenType::AbstractFloatLiteral:
        return "AbstractFloatLiteral"_s;
    case TokenType::IntegerLiteral:
        return "IntegerLiteral"_s;
    case TokenType::IntegerLiteralSigned:
        return "IntegerLiteralSigned"_s;
    case TokenType::IntegerLiteralUnsigned:
        return "IntegerLiteralUnsigned"_s;
    case TokenType::FloatLiteral:
        return "FloatLiteral"_s;
    case TokenType::HalfLiteral:
        return "HalfLiteral"_s;
    case TokenType::Identifier:
        return "Identifier"_s;
    case TokenType::ReservedWord:
        return "ReservedWord"_s;

#define KEYWORD_TO_STRING(lexeme, name) \
    case TokenType::Keyword##name: \
        return #lexeme##_s;
FOREACH_KEYWORD(KEYWORD_TO_STRING)
#undef KEYWORD_TO_STRING

    case TokenType::And:
        return "&"_s;
    case TokenType::AndAnd:
        return "&&"_s;
    case TokenType::AndEq:
        return "&="_s;
    case TokenType::Arrow:
        return "->"_s;
    case TokenType::Attribute:
        return "@"_s;
    case TokenType::Bang:
        return "!"_s;
    case TokenType::BangEq:
        return "!="_s;
    case TokenType::BracketLeft:
        return "["_s;
    case TokenType::BracketRight:
        return "]"_s;
    case TokenType::BraceLeft:
        return "{"_s;
    case TokenType::BraceRight:
        return "}"_s;
    case TokenType::Colon:
        return ":"_s;
    case TokenType::Comma:
        return ","_s;
    case TokenType::Equal:
        return "="_s;
    case TokenType::EqEq:
        return "=="_s;
    case TokenType::TemplateArgsRight:
    case TokenType::Gt:
        return ">"_s;
    case TokenType::GtEq:
        return ">="_s;
    case TokenType::GtGt:
        return ">>"_s;
    case TokenType::GtGtEq:
        return ">>="_s;
    case TokenType::TemplateArgsLeft:
    case TokenType::Lt:
        return "<"_s;
    case TokenType::LtEq:
        return "<="_s;
    case TokenType::LtLt:
        return "<<"_s;
    case TokenType::LtLtEq:
        return "<<="_s;
    case TokenType::Minus:
        return "-"_s;
    case TokenType::MinusMinus:
        return "--"_s;
    case TokenType::MinusEq:
        return "-="_s;
    case TokenType::Modulo:
        return "%"_s;
    case TokenType::ModuloEq:
        return "%="_s;
    case TokenType::Or:
        return "|"_s;
    case TokenType::OrOr:
        return "||"_s;
    case TokenType::OrEq:
        return "|="_s;
    case TokenType::Plus:
        return "+"_s;
    case TokenType::PlusPlus:
        return "++"_s;
    case TokenType::PlusEq:
        return "+="_s;
    case TokenType::Period:
        return "."_s;
    case TokenType::ParenLeft:
        return "("_s;
    case TokenType::ParenRight:
        return ")"_s;
    case TokenType::Semicolon:
        return ";"_s;
    case TokenType::Slash:
        return "/"_s;
    case TokenType::SlashEq:
        return "/="_s;
    case TokenType::Star:
        return "*"_s;
    case TokenType::StarEq:
        return "*="_s;
    case TokenType::Tilde:
        return "~"_s;
    case TokenType::Underbar:
        return "_"_s;
    case TokenType::Xor:
        return "^"_s;
    case TokenType::XorEq:
        return "^="_s;
    case TokenType::Placeholder:
        return "<placeholder>"_s;
    }
}

}
