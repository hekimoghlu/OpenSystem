/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 11, 2025.
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

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <wtf/Forward.h>
#include <wtf/MathExtras.h>
#include <wtf/text/LChar.h>

namespace WTF {

enum PositiveOrNegativeNumber { PositiveNumber, NegativeNumber };

template<typename> struct IntegerToStringConversionTrait;

template<typename T, typename UnsignedIntegerType, PositiveOrNegativeNumber NumberType, typename AdditionalArgumentType>
static typename IntegerToStringConversionTrait<T>::ReturnType numberToStringImpl(UnsignedIntegerType number, AdditionalArgumentType additionalArgument)
{
    std::array<LChar, sizeof(UnsignedIntegerType) * 3 + 1> buffer;
    auto index = buffer.size();
    do {
        buffer[--index] = static_cast<LChar>((number % 10) + '0');
        number /= 10;
    } while (number);

    if (NumberType == NegativeNumber)
        buffer[--index] = '-';

    return IntegerToStringConversionTrait<T>::flush(std::span { buffer }.subspan(index), additionalArgument);
}

template<typename T, typename SignedIntegerType>
inline typename IntegerToStringConversionTrait<T>::ReturnType numberToStringSigned(SignedIntegerType number, typename IntegerToStringConversionTrait<T>::AdditionalArgumentType* additionalArgument = nullptr)
{
    if (number < 0)
        return numberToStringImpl<T, typename std::make_unsigned_t<SignedIntegerType>, NegativeNumber>(-static_cast<typename std::make_unsigned_t<SignedIntegerType>>(number), additionalArgument);
    return numberToStringImpl<T, typename std::make_unsigned_t<SignedIntegerType>, PositiveNumber>(number, additionalArgument);
}

template<typename T, typename UnsignedIntegerType>
inline typename IntegerToStringConversionTrait<T>::ReturnType numberToStringUnsigned(UnsignedIntegerType number, typename IntegerToStringConversionTrait<T>::AdditionalArgumentType* additionalArgument = nullptr)
{
    return numberToStringImpl<T, UnsignedIntegerType, PositiveNumber>(number, additionalArgument);
}

template<typename CharacterType, typename UnsignedIntegerType, PositiveOrNegativeNumber NumberType>
static void writeIntegerToBufferImpl(UnsignedIntegerType number, std::span<CharacterType> destination)
{
    static_assert(!std::is_same_v<bool, std::remove_cv_t<UnsignedIntegerType>>, "'bool' not supported");
    std::array<LChar, sizeof(UnsignedIntegerType) * 3 + 1> buffer;
    auto index = buffer.size();
    do {
        buffer[--index] = static_cast<LChar>((number % 10) + '0');
        number /= 10;
    } while (number);

    if (NumberType == NegativeNumber)
        buffer[--index] = '-';
    
    for (size_t i = 0; i < buffer.size() - index; ++i)
        destination[i] = static_cast<CharacterType>(buffer[index + i]);
}

template<typename CharacterType, typename IntegerType>
inline void writeIntegerToBuffer(IntegerType integer, std::span<CharacterType> destination)
{
    static_assert(std::is_integral_v<IntegerType>);
    if constexpr (std::is_same_v<IntegerType, bool>)
        return writeIntegerToBufferImpl<CharacterType, uint8_t, PositiveNumber>(integer ? 1 : 0, destination);
    else if constexpr (std::is_signed_v<IntegerType>) {
        if (integer < 0)
            return writeIntegerToBufferImpl<CharacterType, typename std::make_unsigned_t<IntegerType>, NegativeNumber>(WTF::negate(integer), destination);
        return writeIntegerToBufferImpl<CharacterType, typename std::make_unsigned_t<IntegerType>, PositiveNumber>(std::make_unsigned_t<IntegerType>(integer), destination);
    } else
        return writeIntegerToBufferImpl<CharacterType, IntegerType, PositiveNumber>(integer, destination);
}

template<typename UnsignedIntegerType, PositiveOrNegativeNumber NumberType>
constexpr unsigned lengthOfIntegerAsStringImpl(UnsignedIntegerType number)
{
    unsigned length = 0;

    do {
        ++length;
        number /= 10;
    } while (number);

    if (NumberType == NegativeNumber)
        ++length;

    return length;
}

template<typename IntegerType>
constexpr unsigned lengthOfIntegerAsString(IntegerType integer)
{
    static_assert(std::is_integral_v<IntegerType>);
    if constexpr (std::is_same_v<IntegerType, bool>) {
        UNUSED_PARAM(integer);
        return 1;
    }
    else if constexpr (std::is_signed_v<IntegerType>) {
        if (integer < 0)
            return lengthOfIntegerAsStringImpl<typename std::make_unsigned_t<IntegerType>, NegativeNumber>(WTF::negate(integer));
        return lengthOfIntegerAsStringImpl<typename std::make_unsigned_t<IntegerType>, PositiveNumber>(std::make_unsigned_t<IntegerType>(integer));
    } else
        return lengthOfIntegerAsStringImpl<IntegerType, PositiveNumber>(integer);
}

template<size_t N>
struct IntegerToStringConversionTrait<Vector<LChar, N>> {
    using ReturnType = Vector<LChar, N>;
    using AdditionalArgumentType = void;
    static ReturnType flush(std::span<const LChar> characters, void*) { return characters; }
};

} // namespace WTF

using WTF::numberToStringSigned;
using WTF::numberToStringUnsigned;
using WTF::lengthOfIntegerAsString;
using WTF::writeIntegerToBuffer;
