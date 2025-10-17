/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 30, 2024.
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

#include <span>
#include <wtf/EnumTraits.h>
#include <wtf/SHA1.h>
#include <wtf/StdLibExtras.h>
#include <wtf/Vector.h>
#include <wtf/persistence/PersistentCoders.h>

namespace WTF::Persistence {

template<typename> struct Coder;

class Encoder {
    WTF_MAKE_FAST_ALLOCATED;
public:
    WTF_EXPORT_PRIVATE Encoder();
    WTF_EXPORT_PRIVATE ~Encoder();

    WTF_EXPORT_PRIVATE void encodeChecksum();
    WTF_EXPORT_PRIVATE void encodeFixedLengthData(std::span<const uint8_t>);

    template<typename T, std::enable_if_t<std::is_enum<T>::value>* = nullptr>
    Encoder& operator<<(const T& t)
    {
        static_assert(sizeof(T) <= sizeof(uint64_t), "Enum type must not be larger than 64 bits.");
        return *this << static_cast<uint64_t>(t);
    }

    template<typename T, std::enable_if_t<!std::is_enum<T>::value && !std::is_arithmetic<typename std::remove_const<T>>::value>* = nullptr>
    Encoder& operator<<(const T& t)
    {
        Coder<T>::encodeForPersistence(*this, t);
        return *this;
    }

    WTF_EXPORT_PRIVATE Encoder& operator<<(bool);
    WTF_EXPORT_PRIVATE Encoder& operator<<(uint8_t);
    WTF_EXPORT_PRIVATE Encoder& operator<<(uint16_t);
    WTF_EXPORT_PRIVATE Encoder& operator<<(uint32_t);
    WTF_EXPORT_PRIVATE Encoder& operator<<(uint64_t);
    WTF_EXPORT_PRIVATE Encoder& operator<<(int16_t);
    WTF_EXPORT_PRIVATE Encoder& operator<<(int32_t);
    WTF_EXPORT_PRIVATE Encoder& operator<<(int64_t);
    WTF_EXPORT_PRIVATE Encoder& operator<<(float);
    WTF_EXPORT_PRIVATE Encoder& operator<<(double);

    const uint8_t* buffer() const LIFETIME_BOUND { return m_buffer.data(); }
    size_t bufferSize() const { return m_buffer.size(); }
    std::span<const uint8_t> span() const LIFETIME_BOUND { return m_buffer.span(); }

    WTF_EXPORT_PRIVATE static void updateChecksumForData(SHA1&, std::span<const uint8_t>);
    template <typename Type> static void updateChecksumForNumber(SHA1&, Type);

    static constexpr bool isIPCEncoder = false;

private:

    template<typename Type> Encoder& encodeNumber(Type);

    std::span<uint8_t> grow(size_t);

    template <typename Type> struct Salt;

    Vector<uint8_t, 4096> m_buffer;
    SHA1 m_sha1;
};

template <> struct Encoder::Salt<bool> { static constexpr unsigned value = 3; };
template <> struct Encoder::Salt<uint8_t> { static constexpr  unsigned value = 5; };
template <> struct Encoder::Salt<uint16_t> { static constexpr unsigned value = 7; };
template <> struct Encoder::Salt<uint32_t> { static constexpr unsigned value = 11; };
template <> struct Encoder::Salt<uint64_t> { static constexpr unsigned value = 13; };
template <> struct Encoder::Salt<int32_t> { static constexpr unsigned value = 17; };
template <> struct Encoder::Salt<int64_t> { static constexpr unsigned value = 19; };
template <> struct Encoder::Salt<float> { static constexpr unsigned value = 23; };
template <> struct Encoder::Salt<double> { static constexpr unsigned value = 29; };
template <> struct Encoder::Salt<uint8_t*> { static constexpr unsigned value = 101; };
template <> struct Encoder::Salt<int16_t> { static constexpr unsigned value = 103; };

template <typename Type>
void Encoder::updateChecksumForNumber(SHA1& sha1, Type value)
{
    auto typeSalt = Salt<Type>::value;
    sha1.addBytes(asByteSpan(typeSalt));
    sha1.addBytes(asByteSpan(value));
}

}
