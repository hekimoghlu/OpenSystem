/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 13, 2023.
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

#include "DaemonCoders.h"

namespace WebKit::Daemon {

class Decoder {
public:
    Decoder(std::span<const uint8_t> buffer)
        : m_buffer(buffer) { }
    ~Decoder();

    template<typename T>
    Decoder& operator>>(std::optional<T>& t)
    {
        t = decode<T>();
        return *this;
    }

    template<typename T>
    std::optional<T> decode()
    {
        return Coder<std::remove_cvref_t<T>>::decode(*this);
    }

    template<typename T>
    WARN_UNUSED_RETURN bool bufferIsLargeEnoughToContain(size_t numElements) const
    {
        static_assert(std::is_arithmetic<T>::value, "Type T must have a fixed, known encoded size!");

        if (numElements > std::numeric_limits<size_t>::max() / sizeof(T))
            return false;

        return bufferIsLargeEnoughToContainBytes(numElements * sizeof(T));
    }

    WARN_UNUSED_RETURN bool decodeFixedLengthData(std::span<uint8_t> data);
    std::span<const uint8_t> decodeFixedLengthReference(size_t);

private:
    WARN_UNUSED_RETURN bool bufferIsLargeEnoughToContainBytes(size_t) const;

    std::span<const uint8_t> m_buffer;
    size_t m_bufferPosition { 0 };
};

} // namespace WebKit::Daemon
