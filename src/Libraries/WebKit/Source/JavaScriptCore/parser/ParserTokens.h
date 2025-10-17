/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 19, 2025.
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

#include <limits.h>
#include <stdint.h>

namespace WTF {
class PrintStream;
}

namespace JSC {

class Identifier;

#define BINARY_OP_PRECEDENCE(prec) (((prec) << BinaryOpTokenPrecedenceShift) | ((prec) << (BinaryOpTokenPrecedenceShift + BinaryOpTokenAllowsInPrecedenceAdditionalShift)))
#define IN_OP_PRECEDENCE(prec) ((prec) << (BinaryOpTokenPrecedenceShift + BinaryOpTokenAllowsInPrecedenceAdditionalShift))

enum JSTokenType {
    // Token Bitfield: 0b000000000RTE00IIIIPPPPKUXXXXXXXX
    // R = right-associative bit
    // T = unterminated error flag
    // E = error flag
    // I = binary operator allows 'in'
    // P = binary operator precedence
    // K = keyword flag
    // U = unary operator flag
    //
    // We must keep the upper 8bit (1byte) region empty. JSTokenType must be 24bits.
    UnaryOpTokenFlag = 1 << 8,
    KeywordTokenFlag = 1 << 9,
    BinaryOpTokenPrecedenceShift = 10,
    BinaryOpTokenAllowsInPrecedenceAdditionalShift = 4,
    BinaryOpTokenPrecedenceMask = 15 << BinaryOpTokenPrecedenceShift,
    CanBeErrorTokenFlag = 1 << (BinaryOpTokenAllowsInPrecedenceAdditionalShift + BinaryOpTokenPrecedenceShift + 6),
    UnterminatedCanBeErrorTokenFlag = CanBeErrorTokenFlag << 1,
    RightAssociativeBinaryOpTokenFlag = UnterminatedCanBeErrorTokenFlag << 1,


    NULLTOKEN = KeywordTokenFlag,
    TRUETOKEN,
    FALSETOKEN,
    BREAK,
    CASE,
    DEFAULT,
    FOR,
    NEW,
    VAR,
    CONSTTOKEN,
    CONTINUE,
    FUNCTION,
    RETURN,
    IF,
    THISTOKEN,
    DO,
    WHILE,
    SWITCH,
    WITH,
    RESERVED,
    RESERVED_IF_STRICT,
    THROW,
    TRY,
    CATCH,
    FINALLY,
    DEBUGGER,
    ELSE,
    IMPORT,
    EXPORT_,
    CLASSTOKEN,
    EXTENDS,
    SUPER,

    // Contextual keywords
    
    LET,
    YIELD,
    AWAIT,

    FirstContextualKeywordToken = LET,
    LastContextualKeywordToken = AWAIT,

    OPENBRACE = 0,
    CLOSEBRACE,
    OPENPAREN,
    CLOSEPAREN,
    OPENBRACKET,
    CLOSEBRACKET,
    COMMA,
    QUESTION,
    BACKQUOTE,
    INTEGER,
    DOUBLE,
    BIGINT,
    IDENT,
    PRIVATENAME,
    STRING,
    TEMPLATE,
    REGEXP,
    SEMICOLON,
    COLON,
    DOT,
    EOFTOK,
    EQUAL,
    PLUSEQUAL,
    MINUSEQUAL,
    MULTEQUAL,
    DIVEQUAL,
    LSHIFTEQUAL,
    RSHIFTEQUAL,
    URSHIFTEQUAL,
    MODEQUAL,
    POWEQUAL,
    BITANDEQUAL,
    BITXOREQUAL,
    BITOREQUAL,
    COALESCEEQUAL,
    OREQUAL,
    ANDEQUAL,
    DOTDOTDOT,
    ARROWFUNCTION,
    QUESTIONDOT,
    LastUntaggedToken,

    // Begin tagged tokens
    PLUSPLUS = 0 | UnaryOpTokenFlag,
    MINUSMINUS = 1 | UnaryOpTokenFlag,
    AUTOPLUSPLUS = 2 | UnaryOpTokenFlag,
    AUTOMINUSMINUS = 3 | UnaryOpTokenFlag,
    EXCLAMATION = 4 | UnaryOpTokenFlag,
    TILDE = 5 | UnaryOpTokenFlag,
    TYPEOF = 6 | UnaryOpTokenFlag | KeywordTokenFlag,
    VOIDTOKEN = 7 | UnaryOpTokenFlag | KeywordTokenFlag,
    DELETETOKEN = 8 | UnaryOpTokenFlag | KeywordTokenFlag,
    COALESCE = 0 | BINARY_OP_PRECEDENCE(1),
    OR = 0 | BINARY_OP_PRECEDENCE(2),
    AND = 0 | BINARY_OP_PRECEDENCE(3),
    BITOR = 0 | BINARY_OP_PRECEDENCE(4),
    BITXOR = 0 | BINARY_OP_PRECEDENCE(5),
    BITAND = 0 | BINARY_OP_PRECEDENCE(6),
    EQEQ = 0 | BINARY_OP_PRECEDENCE(7),
    NE = 1 | BINARY_OP_PRECEDENCE(7),
    STREQ = 2 | BINARY_OP_PRECEDENCE(7),
    STRNEQ = 3 | BINARY_OP_PRECEDENCE(7),
    LT = 0 | BINARY_OP_PRECEDENCE(8),
    GT = 1 | BINARY_OP_PRECEDENCE(8),
    LE = 2 | BINARY_OP_PRECEDENCE(8),
    GE = 3 | BINARY_OP_PRECEDENCE(8),
    INSTANCEOF = 4 | BINARY_OP_PRECEDENCE(8) | KeywordTokenFlag,
    INTOKEN = 5 | IN_OP_PRECEDENCE(8) | KeywordTokenFlag,
    LSHIFT = 0 | BINARY_OP_PRECEDENCE(9),
    RSHIFT = 1 | BINARY_OP_PRECEDENCE(9),
    URSHIFT = 2 | BINARY_OP_PRECEDENCE(9),
    PLUS = 0 | BINARY_OP_PRECEDENCE(10) | UnaryOpTokenFlag,
    MINUS = 1 | BINARY_OP_PRECEDENCE(10) | UnaryOpTokenFlag,
    TIMES = 0 | BINARY_OP_PRECEDENCE(11),
    DIVIDE = 1 | BINARY_OP_PRECEDENCE(11),
    MOD = 2 | BINARY_OP_PRECEDENCE(11),
    POW = 0 | BINARY_OP_PRECEDENCE(12) | RightAssociativeBinaryOpTokenFlag, // Make sure that POW has the highest operator precedence.
    ERRORTOK = 0 | CanBeErrorTokenFlag,
    UNTERMINATED_IDENTIFIER_ESCAPE_ERRORTOK = 0 | CanBeErrorTokenFlag | UnterminatedCanBeErrorTokenFlag,
    INVALID_IDENTIFIER_ESCAPE_ERRORTOK = 1 | CanBeErrorTokenFlag,
    UNTERMINATED_IDENTIFIER_UNICODE_ESCAPE_ERRORTOK = 2 | CanBeErrorTokenFlag | UnterminatedCanBeErrorTokenFlag,
    INVALID_IDENTIFIER_UNICODE_ESCAPE_ERRORTOK = 3 | CanBeErrorTokenFlag,
    UNTERMINATED_MULTILINE_COMMENT_ERRORTOK = 4 | CanBeErrorTokenFlag | UnterminatedCanBeErrorTokenFlag,
    UNTERMINATED_NUMERIC_LITERAL_ERRORTOK = 5 | CanBeErrorTokenFlag | UnterminatedCanBeErrorTokenFlag,
    UNTERMINATED_OCTAL_NUMBER_ERRORTOK = 6 | CanBeErrorTokenFlag | UnterminatedCanBeErrorTokenFlag,
    INVALID_NUMERIC_LITERAL_ERRORTOK = 7 | CanBeErrorTokenFlag,
    UNTERMINATED_STRING_LITERAL_ERRORTOK = 8 | CanBeErrorTokenFlag | UnterminatedCanBeErrorTokenFlag,
    INVALID_STRING_LITERAL_ERRORTOK = 9 | CanBeErrorTokenFlag,
    INVALID_PRIVATE_NAME_ERRORTOK = 10 | CanBeErrorTokenFlag,
    UNTERMINATED_HEX_NUMBER_ERRORTOK = 11 | CanBeErrorTokenFlag | UnterminatedCanBeErrorTokenFlag,
    UNTERMINATED_BINARY_NUMBER_ERRORTOK = 12 | CanBeErrorTokenFlag | UnterminatedCanBeErrorTokenFlag,
    UNTERMINATED_TEMPLATE_LITERAL_ERRORTOK = 13 | CanBeErrorTokenFlag | UnterminatedCanBeErrorTokenFlag,
    UNTERMINATED_REGEXP_LITERAL_ERRORTOK = 14 | CanBeErrorTokenFlag | UnterminatedCanBeErrorTokenFlag,
    INVALID_TEMPLATE_LITERAL_ERRORTOK = 15 | CanBeErrorTokenFlag,
    ESCAPED_KEYWORD = 16 | CanBeErrorTokenFlag,
    INVALID_UNICODE_ENCODING_ERRORTOK = 17 | CanBeErrorTokenFlag,
    INVALID_IDENTIFIER_UNICODE_ERRORTOK = 18 | CanBeErrorTokenFlag,
};
static_assert(static_cast<unsigned>(POW) <= 0x00ffffffU, "JSTokenType must be 24bits.");

struct JSTextPosition {
    JSTextPosition() = default;
    JSTextPosition(int _line, int _offset, int _lineStartOffset) 
        : line(_line)
        , offset(_offset)
        , lineStartOffset(_lineStartOffset)
    { 
        checkConsistency();
    }

    JSTextPosition operator+(int adjustment) const { return JSTextPosition(line, offset + adjustment, lineStartOffset); }
    JSTextPosition operator+(unsigned adjustment) const { return *this + static_cast<int>(adjustment); }
    JSTextPosition operator-(int adjustment) const { return *this + (- adjustment); }
    JSTextPosition operator-(unsigned adjustment) const { return *this + (- static_cast<int>(adjustment)); }

    operator int() const { return offset; }
    explicit operator bool() const { return *this != JSTextPosition(); }

    friend bool operator==(const JSTextPosition&, const JSTextPosition&) = default;

    int column() const { return offset - lineStartOffset; }
    void checkConsistency()
    {
        // FIXME: We should test ASSERT(offset >= lineStartOffset); but that breaks a lot of tests.
        ASSERT(line >= 0);
        ASSERT(offset >= 0);
        ASSERT(lineStartOffset >= 0);
    }

    // FIXME: these should be unsigned.
    int line { -1 };
    int offset { -1 };
    int lineStartOffset { -1 };
};

union JSTokenData {
    struct {
        const Identifier* cooked;
        const Identifier* raw;
        bool isTail;
    };
    struct {
        uint32_t line;
        uint32_t offset;
        uint32_t lineStartOffset;
    };
    double doubleValue;
    struct {
        const Identifier* ident;
        bool escaped;
    };
    struct {
        const Identifier* bigIntString;
        uint8_t radix;
    };
    struct {
        const Identifier* pattern;
        const Identifier* flags;
    };
};

struct JSTokenLocation {
    JSTokenLocation() = default;

    int line { 0 };
    unsigned lineStartOffset { 0 };
    unsigned startOffset { 0 };
    unsigned endOffset { 0 };
};

struct JSToken {
    JSTokenType m_type { ERRORTOK };
    JSTokenData m_data { { nullptr, nullptr, false } };
    JSTokenLocation m_location;
    JSTextPosition m_startPosition;
    JSTextPosition m_endPosition;

    void dump(WTF::PrintStream&) const;
};

ALWAYS_INLINE bool isUpdateOp(JSTokenType token)
{
    return token >= PLUSPLUS && token <= AUTOMINUSMINUS;
}

ALWAYS_INLINE bool isUnaryOp(JSTokenType token)
{
    return token & UnaryOpTokenFlag;
}

} // namespace JSC
