/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 12, 2024.
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

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace WTF {

#if CPU(NEEDS_ALIGNED_ACCESS)

#define COMPARE_2CHARS(address, char1, char2) \
(((address)[0] == char1) && ((address)[1] == char2))
#define COMPARE_4CHARS(address, char1, char2, char3, char4) \
    (COMPARE_2CHARS(address, char1, char2) && COMPARE_2CHARS((address) + 2, char3, char4))
#define COMPARE_2UCHARS(address, char1, char2) \
    (((address)[0] == char1) && ((address)[1] == char2))
#define COMPARE_4UCHARS(address, char1, char2, char3, char4) \
    (COMPARE_2UCHARS(address, char1, char2) && COMPARE_2UCHARS((address) + 2, char3, char4))

#else // CPU(NEEDS_ALIGNED_ACCESS)

#if CPU(BIG_ENDIAN)

#define CHARPAIR_TOUINT16(a, b) \
((((uint16_t)(a)) << 8) + (uint16_t)(b))
#define CHARQUAD_TOUINT32(a, b, c, d) \
    ((((uint32_t)(CHARPAIR_TOUINT16(a, b))) << 16) + CHARPAIR_TOUINT16(c, d))
#define UCHARPAIR_TOUINT32(a, b) \
    ((((uint32_t)(a)) << 16) + (uint32_t)(b))
#define UCHARQUAD_TOUINT64(a, b, c, d) \
    ((((uint64_t)(UCHARQUAD_TOUINT64(a, b))) << 32) + UCHARPAIR_TOUINT32(c, d))

#else // CPU(BIG_ENDIAN)

#define CHARPAIR_TOUINT16(a, b) \
((((uint16_t)(b)) << 8) + (uint16_t)(a))
#define CHARQUAD_TOUINT32(a, b, c, d) \
    ((((uint32_t)(CHARPAIR_TOUINT16(c, d))) << 16) + CHARPAIR_TOUINT16(a, b))
#define UCHARPAIR_TOUINT32(a, b) \
    ((((uint32_t)(b)) << 16) + (uint32_t)(a))
#define UCHARQUAD_TOUINT64(a, b, c, d) \
    ((((uint64_t)(UCHARPAIR_TOUINT32(c, d))) << 32) + UCHARPAIR_TOUINT32(a, b))

#endif // CPU(BIG_ENDIAN)


#define COMPARE_2CHARS(address, char1, char2) \
    ((reinterpret_cast_ptr<const uint16_t*>(address))[0] == CHARPAIR_TOUINT16(char1, char2))
#define COMPARE_2UCHARS(address, char1, char2) \
    ((reinterpret_cast_ptr<const uint32_t*>(address))[0] == UCHARPAIR_TOUINT32(char1, char2))

#if CPU(X86_64) || CPU(ARM64)

#define COMPARE_4CHARS(address, char1, char2, char3, char4) \
    ((reinterpret_cast_ptr<const uint32_t*>(address))[0] == CHARQUAD_TOUINT32(char1, char2, char3, char4))
#define COMPARE_4UCHARS(address, char1, char2, char3, char4) \
    ((reinterpret_cast_ptr<const uint64_t*>(address))[0] == UCHARQUAD_TOUINT64(char1, char2, char3, char4))

#else // CPU(X86_64) || CPU(ARM64)

#define COMPARE_4CHARS(address, char1, char2, char3, char4) \
    (COMPARE_2CHARS(address, char1, char2) && COMPARE_2CHARS((address) + 2, char3, char4))
#define COMPARE_4UCHARS(address, char1, char2, char3, char4) \
    (COMPARE_2UCHARS(address, char1, char2) && COMPARE_2UCHARS((address) + 2, char3, char4))

#endif // CPU(X86_64) || CPU(ARM64)

#endif // CPU(NEEDS_ALIGNED_ACCESS)

#define COMPARE_3CHARS(address, char1, char2, char3) \
    (COMPARE_2CHARS(address, char1, char2) && ((address)[2] == (char3)))
#define COMPARE_3UCHARS(address, char1, char2, char3) \
    (COMPARE_2UCHARS(address, char1, char2) && ((address)[2] == (char3)))
#define COMPARE_5CHARS(address, char1, char2, char3, char4, char5) \
    (COMPARE_4CHARS(address, char1, char2, char3, char4) && ((address)[4] == (char5)))
#define COMPARE_5UCHARS(address, char1, char2, char3, char4, char5) \
    (COMPARE_4UCHARS(address, char1, char2, char3, char4) && ((address)[4] == (char5)))
#define COMPARE_6CHARS(address, char1, char2, char3, char4, char5, char6) \
    (COMPARE_4CHARS(address, char1, char2, char3, char4) && COMPARE_2CHARS(address + 4, char5, char6))
#define COMPARE_6UCHARS(address, char1, char2, char3, char4, char5, char6) \
    (COMPARE_4UCHARS(address, char1, char2, char3, char4) && COMPARE_2UCHARS(address + 4, char5, char6))
#define COMPARE_7CHARS(address, char1, char2, char3, char4, char5, char6, char7) \
    (COMPARE_4CHARS(address, char1, char2, char3, char4) && COMPARE_4CHARS(address + 3, char4, char5, char6, char7))
#define COMPARE_7UCHARS(address, char1, char2, char3, char4, char5, char6, char7) \
    (COMPARE_4UCHARS(address, char1, char2, char3, char4) && COMPARE_4UCHARS(address + 3, char4, char5, char6, char7))
#define COMPARE_8CHARS(address, char1, char2, char3, char4, char5, char6, char7, char8) \
    (COMPARE_4CHARS(address, char1, char2, char3, char4) && COMPARE_4CHARS(address + 4, char5, char6, char7, char8))
#define COMPARE_8UCHARS(address, char1, char2, char3, char4, char5, char6, char7, char8) \
    (COMPARE_4UCHARS(address, char1, char2, char3, char4) && COMPARE_4UCHARS(address + 4, char5, char6, char7, char8))

template<typename CharacterType>
ALWAYS_INLINE static bool compareCharacters(const CharacterType* source, char c0, char c1)
{
    if constexpr (sizeof(CharacterType) == 1)
        return COMPARE_2CHARS(source, c0, c1);
    else
        return COMPARE_2UCHARS(source, c0, c1);
}

template<typename CharacterType>
ALWAYS_INLINE static bool compareCharacters(const CharacterType* source, char c0, char c1, char c2)
{
    if constexpr (sizeof(CharacterType) == 1)
        return COMPARE_3CHARS(source, c0, c1, c2);
    else
        return COMPARE_3UCHARS(source, c0, c1, c2);
}

template<typename CharacterType>
ALWAYS_INLINE static bool compareCharacters(const CharacterType* source, char c0, char c1, char c2, char c3)
{
    if constexpr (sizeof(CharacterType) == 1)
        return COMPARE_4CHARS(source, c0, c1, c2, c3);
    else
        return COMPARE_4UCHARS(source, c0, c1, c2, c3);
}

template<typename CharacterType>
ALWAYS_INLINE static bool compareCharacters(const CharacterType* source, char c0, char c1, char c2, char c3, char c4)
{
    if constexpr (sizeof(CharacterType) == 1)
        return COMPARE_5CHARS(source, c0, c1, c2, c3, c4);
    else
        return COMPARE_5UCHARS(source, c0, c1, c2, c3, c4);
}

template<typename CharacterType>
ALWAYS_INLINE static bool compareCharacters(const CharacterType* source, char c0, char c1, char c2, char c3, char c4, char c5)
{
    if constexpr (sizeof(CharacterType) == 1)
        return COMPARE_6CHARS(source, c0, c1, c2, c3, c4, c5);
    else
        return COMPARE_6UCHARS(source, c0, c1, c2, c3, c4, c5);
}

template<typename CharacterType>
ALWAYS_INLINE static bool compareCharacters(const CharacterType* source, char c0, char c1, char c2, char c3, char c4, char c5, char c6)
{
    if constexpr (sizeof(CharacterType) == 1)
        return COMPARE_7CHARS(source, c0, c1, c2, c3, c4, c5, c6);
    else
        return COMPARE_7UCHARS(source, c0, c1, c2, c3, c4, c5, c6);
}

template<typename CharacterType>
ALWAYS_INLINE static bool compareCharacters(const CharacterType* source, char c0, char c1, char c2, char c3, char c4, char c5, char c6, char c7)
{
    if constexpr (sizeof(CharacterType) == 1)
        return COMPARE_8CHARS(source, c0, c1, c2, c3, c4, c5, c6, c7);
    else
        return COMPARE_8UCHARS(source, c0, c1, c2, c3, c4, c5, c6, c7);
}

#undef COMPARE_2CHARS
#undef COMPARE_2UCHARS
#undef COMPARE_3CHARS
#undef COMPARE_3UCHARS
#undef COMPARE_4CHARS
#undef COMPARE_4UCHARS
#undef COMPARE_5CHARS
#undef COMPARE_5UCHARS
#undef COMPARE_6CHARS
#undef COMPARE_6UCHARS
#undef COMPARE_7CHARS
#undef COMPARE_7UCHARS
#undef COMPARE_8CHARS
#undef COMPARE_8UCHARS
#undef CHARPAIR_TOUINT16
#undef CHARQUAD_TOUINT32
#undef UCHARPAIR_TOUINT32
#undef UCHARQUAD_TOUINT64

} // namespace WTF

using WTF::compareCharacters;

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
