/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 10, 2025.
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

#include <wtf/Compiler.h>

#include <wtf/dtoa.h>
#include <wtf/text/IntegerToStringConversion.h>
#include <wtf/text/StringConcatenate.h>

namespace WTF {

template<typename Integer>
class StringTypeAdapter<Integer, typename std::enable_if_t<std::is_integral_v<Integer>>> {
public:
    StringTypeAdapter(Integer number)
        : m_number { number }
    {
    }

    unsigned length() const { return lengthOfIntegerAsString(m_number); }
    bool is8Bit() const { return true; }
    template<typename CharacterType>
    void writeTo(std::span<CharacterType> destination) const { writeIntegerToBuffer(m_number, destination); }

private:
    Integer m_number;
};

template<typename Enum>
class StringTypeAdapter<Enum, typename std::enable_if_t<std::is_enum_v<Enum>>> {
using UnderlyingType = typename std::underlying_type_t<Enum>;
public:
    StringTypeAdapter(Enum enumValue)
        : m_enum { enumValue }
    {
    }

    unsigned length() const { return lengthOfIntegerAsString(static_cast<UnderlyingType>(m_enum)); }
    bool is8Bit() const { return true; }
    template<typename CharacterType>
    void writeTo(std::span<CharacterType> destination) const { writeIntegerToBuffer(static_cast<UnderlyingType>(m_enum), destination); }

private:
    Enum m_enum;
};

template<typename FloatingPoint>
class StringTypeAdapter<FloatingPoint, typename std::enable_if_t<std::is_floating_point<FloatingPoint>::value>> {
public:
    StringTypeAdapter(FloatingPoint number)
    {
        m_length = numberToStringAndSize(number, m_buffer).size();
    }

    unsigned length() const { return m_length; }
    bool is8Bit() const { return true; }
    template<typename CharacterType> void writeTo(std::span<CharacterType> destination) const { StringImpl::copyCharacters(destination, span()); }

private:
    std::span<const LChar> span() const LIFETIME_BOUND { return byteCast<LChar>(std::span { m_buffer }).first(m_length); }

    NumberToStringBuffer m_buffer;
    unsigned m_length;
};

class FormattedNumber {
    WTF_MAKE_FAST_ALLOCATED;
public:
    static FormattedNumber fixedPrecision(double number, unsigned significantFigures = 6, TrailingZerosPolicy trailingZerosTruncatingPolicy = TrailingZerosPolicy::Truncate)
    {
        FormattedNumber numberFormatter;
        numberFormatter.m_length = numberToFixedPrecisionString(number, significantFigures, numberFormatter.m_buffer, trailingZerosTruncatingPolicy == TrailingZerosPolicy::Truncate).size();
        return numberFormatter;
    }

    static FormattedNumber fixedWidth(double number, unsigned decimalPlaces)
    {
        FormattedNumber numberFormatter;
        numberFormatter.m_length = numberToFixedWidthString(number, decimalPlaces, numberFormatter.m_buffer).size();
        return numberFormatter;
    }

    unsigned length() const { return m_length; }
    const LChar* buffer() const LIFETIME_BOUND { return byteCast<LChar>(&m_buffer[0]); }
    std::span<const LChar> span() const LIFETIME_BOUND { return byteCast<LChar>(std::span { m_buffer }).first(m_length); }

private:
    NumberToStringBuffer m_buffer;
    unsigned m_length;
};

template<> class StringTypeAdapter<FormattedNumber> {
public:
    StringTypeAdapter(const FormattedNumber& number)
        : m_number { number }
    {
    }

    unsigned length() const { return m_number.length(); }
    bool is8Bit() const { return true; }
    template<typename CharacterType> void writeTo(std::span<CharacterType> destination) const { StringImpl::copyCharacters(destination, m_number.span()); }

private:
    const FormattedNumber& m_number;
};

class FormattedCSSNumber {
    WTF_MAKE_FAST_ALLOCATED;
public:
    static FormattedCSSNumber create(double number)
    {
        FormattedCSSNumber numberFormatter;
        numberFormatter.m_length = numberToCSSString(number, numberFormatter.m_buffer).size();
        return numberFormatter;
    } 

    unsigned length() const { return m_length; }
    const LChar* buffer() const LIFETIME_BOUND { return byteCast<LChar>(&m_buffer[0]); }
    std::span<const LChar> span() const LIFETIME_BOUND { return byteCast<LChar>(std::span { m_buffer }).first(m_length); }

private:
    NumberToCSSStringBuffer m_buffer;
    unsigned m_length;
};

template<> class StringTypeAdapter<FormattedCSSNumber> {
public:
    StringTypeAdapter(const FormattedCSSNumber& number)
        : m_number { number }
    {
    }

    unsigned length() const { return m_number.length(); }
    bool is8Bit() const { return true; }
    template<typename CharacterType> void writeTo(std::span<CharacterType> destination) const { StringImpl::copyCharacters(destination, m_number.span()); }

private:
    const FormattedCSSNumber& m_number;
};

}

using WTF::FormattedNumber;
using WTF::FormattedCSSNumber;
