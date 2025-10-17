/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 24, 2023.
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

#include <optional>
#include <span>
#include <wtf/EnumTraits.h>

namespace IPC {

class Decoder;
class Encoder;

template<typename T, typename = void> struct ArgumentCoder;

template<>
struct ArgumentCoder<bool> {
    template<typename Encoder>
    static void encode(Encoder& encoder, bool value)
    {
        uint8_t data = value ? 1 : 0;
        encoder << data;
    }

    template<typename Decoder>
    static std::optional<bool> decode(Decoder& decoder)
    {
        auto data = decoder.template decode<uint8_t>();
        if (data && *data <= 1) // This ensures that only the lower bit is set in a boolean for IPC messages
            return !!*data;
        return std::nullopt;
    }
};

template<typename T>
struct ArgumentCoder<T, typename std::enable_if_t<std::is_arithmetic_v<T>>> {
    template<typename Encoder>
    static void encode(Encoder& encoder, T value)
    {
        encoder.encodeObject(value);
    }

    template<typename Decoder>
    static std::optional<T> decode(Decoder& decoder)
    {
        return decoder.template decodeObject<T>();
    }
};

template<typename T>
struct ArgumentCoder<T, typename std::enable_if_t<std::is_enum_v<T>>> {
    template<typename Encoder>
    static void encode(Encoder& encoder, T value)
    {
        ASSERT(WTF::isValidEnum<T>(WTF::enumToUnderlyingType<T>(value)));
        encoder << WTF::enumToUnderlyingType<T>(value);
    }

    template<typename Decoder>
    static std::optional<T> decode(Decoder& decoder)
    {
        std::optional<std::underlying_type_t<T>> value;
        decoder >> value;
        if (value && WTF::isValidEnum<T>(*value))
            return static_cast<T>(*value);
        return std::nullopt;
    }
};

}
