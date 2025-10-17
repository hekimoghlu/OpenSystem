/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 21, 2024.
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
#include "DateTimeFormat.h"

#include <wtf/ASCIICType.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

static constexpr std::array lowerCaseToFieldTypeMap {
    DateTimeFormat::FieldTypePeriod, // a
    DateTimeFormat::FieldTypeInvalid, // b
    DateTimeFormat::FieldTypeLocalDayOfWeekStandAlon, // c
    DateTimeFormat::FieldTypeDayOfMonth, // d
    DateTimeFormat::FieldTypeLocalDayOfWeek, // e
    DateTimeFormat::FieldTypeInvalid, // f
    DateTimeFormat::FieldTypeModifiedJulianDay, // g
    DateTimeFormat::FieldTypeHour12, // h
    DateTimeFormat::FieldTypeInvalid, // i
    DateTimeFormat::FieldTypeInvalid, // j
    DateTimeFormat::FieldTypeHour24, // k
    DateTimeFormat::FieldTypeInvalid, // l
    DateTimeFormat::FieldTypeMinute, // m
    DateTimeFormat::FieldTypeInvalid, // n
    DateTimeFormat::FieldTypeInvalid, // o
    DateTimeFormat::FieldTypeInvalid, // p
    DateTimeFormat::FieldTypeQuaterStandAlone, // q
    DateTimeFormat::FieldTypeInvalid, // r
    DateTimeFormat::FieldTypeSecond, // s
    DateTimeFormat::FieldTypeInvalid, // t
    DateTimeFormat::FieldTypeExtendedYear, // u
    DateTimeFormat::FieldTypeNonLocationZone, // v
    DateTimeFormat::FieldTypeWeekOfYear, // w
    DateTimeFormat::FieldTypeInvalid, // x
    DateTimeFormat::FieldTypeYear, // y
    DateTimeFormat::FieldTypeZone, // z
};

static constexpr std::array upperCaseToFieldTypeMap {
    DateTimeFormat::FieldTypeMillisecondsInDay, // A
    DateTimeFormat::FieldTypeInvalid, // B
    DateTimeFormat::FieldTypeInvalid, // C
    DateTimeFormat::FieldTypeDayOfYear, // D
    DateTimeFormat::FieldTypeDayOfWeek, // E
    DateTimeFormat::FieldTypeDayOfWeekInMonth, // F
    DateTimeFormat::FieldTypeEra, // G
    DateTimeFormat::FieldTypeHour23, // H
    DateTimeFormat::FieldTypeInvalid, // I
    DateTimeFormat::FieldTypeInvalid, // J
    DateTimeFormat::FieldTypeHour11, // K
    DateTimeFormat::FieldTypeMonthStandAlone, // L
    DateTimeFormat::FieldTypeMonth, // M
    DateTimeFormat::FieldTypeInvalid, // N
    DateTimeFormat::FieldTypeInvalid, // O
    DateTimeFormat::FieldTypeInvalid, // P
    DateTimeFormat::FieldTypeQuater, // Q
    DateTimeFormat::FieldTypeInvalid, // R
    DateTimeFormat::FieldTypeFractionalSecond, // S
    DateTimeFormat::FieldTypeInvalid, // T
    DateTimeFormat::FieldTypeInvalid, // U
    DateTimeFormat::FieldTypeInvalid, // V
    DateTimeFormat::FieldTypeWeekOfMonth, // W
    DateTimeFormat::FieldTypeInvalid, // X
    DateTimeFormat::FieldTypeYearOfWeekOfYear, // Y
    DateTimeFormat::FieldTypeRFC822Zone, // Z
};

static DateTimeFormat::FieldType mapCharacterToFieldType(const UChar ch)
{
    if (isASCIIUpper(ch))
        return upperCaseToFieldTypeMap[ch - 'A'];

    if (isASCIILower(ch))
        return lowerCaseToFieldTypeMap[ch - 'a'];

    return DateTimeFormat::FieldTypeLiteral;
}

bool DateTimeFormat::parse(const String& source, TokenHandler& tokenHandler)
{
    enum State {
        StateInQuote,
        StateInQuoteQuote,
        StateLiteral,
        StateQuote,
        StateSymbol,
    } state = StateLiteral;

    FieldType fieldType = FieldTypeLiteral;
    StringBuilder literalBuffer;
    int fieldCounter = 0;

    for (unsigned int index = 0; index < source.length(); ++index) {
        const UChar ch = source[index];
        switch (state) {
        case StateInQuote:
            if (ch == '\'') {
                state = StateInQuoteQuote;
                break;
            }

            literalBuffer.append(ch);
            break;

        case StateInQuoteQuote:
            if (ch == '\'') {
                literalBuffer.append('\'');
                state = StateInQuote;
                break;
            }

            fieldType = mapCharacterToFieldType(ch);
            if (fieldType == FieldTypeInvalid)
                return false;

            if (fieldType == FieldTypeLiteral) {
                literalBuffer.append(ch);
                state = StateLiteral;
                break;
            }

            if (literalBuffer.length()) {
                tokenHandler.visitLiteral(literalBuffer.toString());
                literalBuffer.clear();
            }

            fieldCounter = 1;
            state = StateSymbol;
            break;

        case StateLiteral:
            if (ch == '\'') {
                state = StateQuote;
                break;
            }

            fieldType = mapCharacterToFieldType(ch);
            if (fieldType == FieldTypeInvalid)
                return false;

            if (fieldType == FieldTypeLiteral) {
                literalBuffer.append(ch);
                break;
            }

            if (literalBuffer.length()) {
                tokenHandler.visitLiteral(literalBuffer.toString());
                literalBuffer.clear();
            }

            fieldCounter = 1;
            state = StateSymbol;
            break;

        case StateQuote:
            literalBuffer.append(ch);
            state = ch == '\'' ? StateLiteral : StateInQuote;
            break;

        case StateSymbol: {
            ASSERT(fieldType != FieldTypeInvalid);
            ASSERT(fieldType != FieldTypeLiteral);
            ASSERT(literalBuffer.isEmpty());

            FieldType fieldType2 = mapCharacterToFieldType(ch);
            if (fieldType2 == FieldTypeInvalid)
                return false;

            if (fieldType == fieldType2) {
                ++fieldCounter;
                break;
            }

            tokenHandler.visitField(fieldType, fieldCounter);

            if (fieldType2 == FieldTypeLiteral) {
                if (ch == '\'')
                    state = StateQuote;
                else {
                    literalBuffer.append(ch);
                    state = StateLiteral;
                }
                break;
            }

            fieldCounter = 1;
            fieldType = fieldType2;
            break;
        }
        }
    }

    ASSERT(fieldType != FieldTypeInvalid);

    switch (state) {
    case StateLiteral:
    case StateInQuoteQuote:
        if (literalBuffer.length())
            tokenHandler.visitLiteral(literalBuffer.toString());
        return true;

    case StateQuote:
    case StateInQuote:
        if (literalBuffer.length())
            tokenHandler.visitLiteral(literalBuffer.toString());
        return false;

    case StateSymbol:
        ASSERT(fieldType != FieldTypeLiteral);
        ASSERT(!literalBuffer.length());
        tokenHandler.visitField(fieldType, fieldCounter);
        return true;
    }

    ASSERT_NOT_REACHED();
    return false;
}

static bool isASCIIAlphabetOrQuote(UChar ch)
{
    return isASCIIAlpha(ch) || ch == '\'';
}

void DateTimeFormat::quoteAndAppendLiteral(const String& literal, StringBuilder& buffer)
{
    if (literal.length() <= 0)
        return;

    if (literal.find(isASCIIAlphabetOrQuote) == notFound) {
        buffer.append(literal);
        return;
    }

    if (literal.find('\'') == notFound) {
        buffer.append('\'', literal, '\'');
        return;
    }

    for (unsigned i = 0; i < literal.length(); ++i) {
        if (literal[i] == '\'')
            buffer.append("''"_s);
        else {
            buffer.append('\'', makeStringByReplacingAll(literal.substring(i), '\'', "''"_s), '\'');
            return;
        }
    }
}

} // namespace WebCore
