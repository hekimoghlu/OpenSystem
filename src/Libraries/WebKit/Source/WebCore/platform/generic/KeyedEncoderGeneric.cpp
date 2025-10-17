/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 26, 2024.
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
#include "KeyedEncoderGeneric.h"

#include "SharedBuffer.h"
#include <wtf/StdLibExtras.h>
#include <wtf/persistence/PersistentEncoder.h>

namespace WebCore {

std::unique_ptr<KeyedEncoder> KeyedEncoder::encoder()
{
    return makeUnique<KeyedEncoderGeneric>();
}

void KeyedEncoderGeneric::encodeString(const String& key)
{
    auto result = key.tryGetUTF8([&](std::span<const char8_t> span) -> bool {
        m_encoder << span.size();
        m_encoder.encodeFixedLengthData(byteCast<uint8_t>(span));
        return true;
    });
    RELEASE_ASSERT(result);
}

void KeyedEncoderGeneric::encodeBytes(const String& key, std::span<const uint8_t> bytes)
{
    m_encoder << Type::Bytes;
    encodeString(key);
    m_encoder << bytes.size();
    m_encoder.encodeFixedLengthData(bytes);
}

void KeyedEncoderGeneric::encodeBool(const String& key, bool value)
{
    m_encoder << Type::Bool;
    encodeString(key);
    m_encoder << value;
}

void KeyedEncoderGeneric::encodeUInt32(const String& key, uint32_t value)
{
    m_encoder << Type::UInt32;
    encodeString(key);
    m_encoder << value;
}

void KeyedEncoderGeneric::encodeUInt64(const String& key, uint64_t value)
{
    m_encoder << Type::UInt64;
    encodeString(key);
    m_encoder << value;
}

void KeyedEncoderGeneric::encodeInt32(const String& key, int32_t value)
{
    m_encoder << Type::Int32;
    encodeString(key);
    m_encoder << value;
}

void KeyedEncoderGeneric::encodeInt64(const String& key, int64_t value)
{
    m_encoder << Type::Int64;
    encodeString(key);
    m_encoder << value;
}

void KeyedEncoderGeneric::encodeFloat(const String& key, float value)
{
    m_encoder << Type::Float;
    encodeString(key);
    m_encoder << value;
}

void KeyedEncoderGeneric::encodeDouble(const String& key, double value)
{
    m_encoder << Type::Double;
    encodeString(key);
    m_encoder << value;
}

void KeyedEncoderGeneric::encodeString(const String& key, const String& value)
{
    m_encoder << Type::String;
    encodeString(key);
    encodeString(value);
}

void KeyedEncoderGeneric::beginObject(const String& key)
{
    m_encoder << Type::BeginObject;
    encodeString(key);
}

void KeyedEncoderGeneric::endObject()
{
    m_encoder << Type::EndObject;
}

void KeyedEncoderGeneric::beginArray(const String& key)
{
    m_encoder << Type::BeginArray;
    encodeString(key);
}

void KeyedEncoderGeneric::beginArrayElement()
{
    m_encoder << Type::BeginArrayElement;
}

void KeyedEncoderGeneric::endArrayElement()
{
    m_encoder << Type::EndArrayElement;
}

void KeyedEncoderGeneric::endArray()
{
    m_encoder << Type::EndArray;
}

RefPtr<SharedBuffer> KeyedEncoderGeneric::finishEncoding()
{
    return SharedBuffer::create(m_encoder.span());
}

} // namespace WebCore
