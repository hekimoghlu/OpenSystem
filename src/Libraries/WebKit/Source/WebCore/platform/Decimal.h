/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 3, 2025.
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
#ifndef Decimal_h
#define Decimal_h

#include <stdint.h>
#include <wtf/Assertions.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

namespace DecimalPrivate {
class SpecialValueHandler;
}

// This class represents decimal base floating point number.
//
// FIXME: Once all C++ compiler support decimal type, we should replace this
// class to compiler supported one. See below URI for current status of decimal
// type for C++: // http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1977.html
class Decimal {
    WTF_MAKE_TZONE_ALLOCATED(Decimal);
public:
    static constexpr int ExponentMax = 1023;
    static constexpr int ExponentMin = -1023;
    static constexpr int Precision = 18;
    static constexpr uint64_t MaxCoefficient = UINT64_C(0XDE0B6B3A763FFFF); // 999999999999999999 == 18 9's

    enum Sign {
        Positive,
        Negative,
    };

    // You should not use EncodedData other than unit testing.
    class EncodedData {
        // For accessing FormatClass.
        friend class Decimal;
        friend class DecimalPrivate::SpecialValueHandler;
    public:
        constexpr EncodedData(Sign, int exponent, uint64_t coefficient);

        friend bool operator==(const EncodedData&, const EncodedData&) = default;

        uint64_t coefficient() const { return m_coefficient; }
        int countDigits() const;
        int exponent() const { return m_exponent; }
        bool isFinite() const { return !isSpecial(); }
        bool isInfinity() const { return m_formatClass == ClassInfinity; }
        bool isNaN() const { return m_formatClass == ClassNaN; }
        bool isSpecial() const { return m_formatClass == ClassInfinity || m_formatClass == ClassNaN; }
        bool isZero() const { return m_formatClass == ClassZero; }
        Sign sign() const { return m_sign; }
        void setSign(Sign sign) { m_sign = sign; }

    private:
        enum FormatClass {
            ClassInfinity,
            ClassNormal,
            ClassNaN,
            ClassZero,
        };

        constexpr EncodedData(Sign, FormatClass);
        FormatClass formatClass() const { return m_formatClass; }

        uint64_t m_coefficient;
        int16_t m_exponent;
        FormatClass m_formatClass;
        Sign m_sign;
    };

    constexpr Decimal(int32_t = 0);
    constexpr Decimal(Sign, int exponent, uint64_t coefficient);

    Decimal& operator+=(const Decimal&);
    Decimal& operator-=(const Decimal&);
    Decimal& operator*=(const Decimal&);
    Decimal& operator/=(const Decimal&);

    Decimal operator-() const;

    bool operator==(const Decimal&) const;
    bool operator!=(const Decimal&) const;
    bool operator<(const Decimal&) const;
    bool operator<=(const Decimal&) const;
    bool operator>(const Decimal&) const;
    bool operator>=(const Decimal&) const;

    Decimal operator+(const Decimal&) const;
    Decimal operator-(const Decimal&) const;
    Decimal operator*(const Decimal&) const;
    Decimal operator/(const Decimal&) const;

    int exponent() const
    {
        ASSERT(isFinite());
        return m_data.exponent();
    }

    bool isFinite() const { return m_data.isFinite(); }
    bool isInfinity() const { return m_data.isInfinity(); }
    bool isNaN() const { return m_data.isNaN(); }
    bool isNegative() const { return sign() == Negative; }
    bool isPositive() const { return sign() == Positive; }
    bool isSpecial() const { return m_data.isSpecial(); }
    bool isZero() const { return m_data.isZero(); }

    Decimal abs() const;
    Decimal ceil() const;
    Decimal floor() const;
    Decimal remainder(const Decimal&) const;
    Decimal round() const;

    double toDouble() const;
    // Note: toString method supports infinity and nan but fromString not.
    String toString() const;

    WEBCORE_EXPORT static Decimal fromDouble(double);
    // fromString supports following syntax EBNF:
    //  number ::= sign? digit+ ('.' digit*) (exponent-marker sign? digit+)?
    //          | sign? '.' digit+ (exponent-marker sign? digit+)?
    //  sign ::= '+' | '-'
    //  exponent-marker ::= 'e' | 'E'
    //  digit ::= '0' | '1' | ... | '9'
    // Note: fromString doesn't support "infinity" and "nan".
    static Decimal fromString(StringView);

    static constexpr Decimal infinity(Sign);
    static constexpr Decimal nan();
    static constexpr Decimal zero(Sign);
    static constexpr Decimal doubleMax();

    // You should not use below methods. We expose them for unit testing.
    explicit constexpr Decimal(const EncodedData& data)
        : m_data(data)
    {
    }

    const EncodedData& value() const { return m_data; }

private:
    struct AlignedOperands {
        uint64_t lhsCoefficient;
        uint64_t rhsCoefficient;
        int exponent;
    };

    Decimal(double);
    WEBCORE_EXPORT Decimal compareTo(const Decimal&) const;

    static AlignedOperands alignOperands(const Decimal& lhs, const Decimal& rhs);
    static inline Sign invertSign(Sign sign) { return sign == Negative ? Positive : Negative; }

    Sign sign() const { return m_data.sign(); }

    EncodedData m_data;
};

inline constexpr Decimal::EncodedData::EncodedData(Sign sign, FormatClass formatClass)
    : m_coefficient(0)
    , m_exponent(0)
    , m_formatClass(formatClass)
    , m_sign(sign)
{
}

inline constexpr Decimal::EncodedData::EncodedData(Sign sign, int exponent, uint64_t coefficient)
    : m_formatClass(coefficient ? ClassNormal : ClassZero)
    , m_sign(sign)
{
    if (exponent >= ExponentMin && exponent <= ExponentMax) {
        while (coefficient > MaxCoefficient) {
            coefficient /= 10;
            ++exponent;
        }
    }

    if (exponent > ExponentMax) {
        m_coefficient = 0;
        m_exponent = 0;
        m_formatClass = ClassInfinity;
        return;
    }

    if (exponent < ExponentMin) {
        m_coefficient = 0;
        m_exponent = 0;
        m_formatClass = ClassZero;
        return;
    }

    m_coefficient = coefficient;
    m_exponent = static_cast<int16_t>(exponent);
}

inline constexpr Decimal::Decimal(int32_t i32)
    : m_data(i32 < 0 ? Negative : Positive, 0, i32 < 0 ? static_cast<uint64_t>(-static_cast<int64_t>(i32)) : static_cast<uint64_t>(i32))
{
}

inline constexpr Decimal::Decimal(Sign sign, int exponent, uint64_t coefficient)
    : m_data(sign, exponent, coefficient)
{
}

inline bool Decimal::operator==(const Decimal& rhs) const
{
    return m_data == rhs.m_data || compareTo(rhs).isZero();
}

inline constexpr Decimal Decimal::infinity(Sign sign)
{
    return Decimal(EncodedData(sign, EncodedData::ClassInfinity));
}

inline constexpr Decimal Decimal::nan()
{
    return Decimal(EncodedData(Positive, EncodedData::ClassNaN));
}

inline constexpr Decimal Decimal::zero(Sign sign)
{
    return Decimal(EncodedData(sign, EncodedData::ClassZero));
}

inline constexpr Decimal Decimal::doubleMax()
{
    return Decimal(EncodedData(Sign::Positive, 292, 17976931348623157));
}

} // namespace WebCore

#endif // Decimal_h
