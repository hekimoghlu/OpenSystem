/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 15, 2023.
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
#include <wtf/persistence/PersistentCoders.h>

namespace WTF::Persistence {

template<typename> struct Coder;

class Decoder {
    WTF_MAKE_FAST_ALLOCATED;
public:
    WTF_EXPORT_PRIVATE Decoder(std::span<const uint8_t>);
    WTF_EXPORT_PRIVATE ~Decoder();

    size_t length() const { return m_buffer.size(); }
    size_t currentOffset() const { return static_cast<size_t>(std::distance(m_buffer.begin(), m_bufferPosition)); }
    
    WTF_EXPORT_PRIVATE WARN_UNUSED_RETURN bool rewind(size_t);

    WTF_EXPORT_PRIVATE WARN_UNUSED_RETURN bool verifyChecksum();

    WTF_EXPORT_PRIVATE WARN_UNUSED_RETURN bool decodeFixedLengthData(std::span<uint8_t>);

    WTF_EXPORT_PRIVATE Decoder& operator>>(std::optional<bool>&);
    WTF_EXPORT_PRIVATE Decoder& operator>>(std::optional<uint8_t>&);
    WTF_EXPORT_PRIVATE Decoder& operator>>(std::optional<uint16_t>&);
    WTF_EXPORT_PRIVATE Decoder& operator>>(std::optional<uint32_t>&);
    WTF_EXPORT_PRIVATE Decoder& operator>>(std::optional<uint64_t>&);
    WTF_EXPORT_PRIVATE Decoder& operator>>(std::optional<int16_t>&);
    WTF_EXPORT_PRIVATE Decoder& operator>>(std::optional<int32_t>&);
    WTF_EXPORT_PRIVATE Decoder& operator>>(std::optional<int64_t>&);
    WTF_EXPORT_PRIVATE Decoder& operator>>(std::optional<float>&);
    WTF_EXPORT_PRIVATE Decoder& operator>>(std::optional<double>&);

    template<typename T, std::enable_if_t<!std::is_arithmetic<typename std::remove_const<T>>::value && !std::is_enum<T>::value>* = nullptr>
    Decoder& operator>>(std::optional<T>& result)
    {
        result = Coder<T>::decodeForPersistence(*this);
        return *this;
    }

    template<typename E, std::enable_if_t<std::is_enum<E>::value>* = nullptr>
    Decoder& operator>>(std::optional<E>& result)
    {
        static_assert(sizeof(E) <= 8, "Enum type T must not be larger than 64 bits!");
        std::optional<uint64_t> value;
        *this >> value;
        if (!value)
            return *this;
        if (!isValidEnumForPersistence<E>(*value))
            return *this;
        result = static_cast<E>(*value);
        return *this;
    }

    template<typename T> WARN_UNUSED_RETURN
    bool bufferIsLargeEnoughToContain(size_t numElements) const
    {
        static_assert(std::is_arithmetic<T>::value, "Type T must have a fixed, known encoded size!");
        return numElements <= std::numeric_limits<size_t>::max() / sizeof(T) && bufferIsLargeEnoughToContain(numElements * sizeof(T));
    }

    WTF_EXPORT_PRIVATE WARN_UNUSED_RETURN std::span<const uint8_t> bufferPointerForDirectRead(size_t numBytes);

private:
    WTF_EXPORT_PRIVATE WARN_UNUSED_RETURN bool bufferIsLargeEnoughToContain(size_t) const;
    template<typename Type> Decoder& decodeNumber(std::optional<Type>&);

    const std::span<const uint8_t> m_buffer;
    std::span<const uint8_t>::iterator m_bufferPosition;

    SHA1 m_sha1;
};

} 
