/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 8, 2022.
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

#include "KeyedCoding.h"
#include <wtf/Forward.h>
#include <wtf/Vector.h>
#include <wtf/persistence/PersistentEncoder.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class FragmentedSharedBuffer;

class KeyedEncoderGeneric final : public KeyedEncoder {
public:

    enum class Type : uint8_t {
        Bytes,
        Bool,
        UInt32,
        UInt64,
        Int32,
        Int64,
        Float,
        Double,
        String,
        BeginObject,
        EndObject,
        BeginArray,
        BeginArrayElement,
        EndArrayElement,
        EndArray,
    };

private:
    RefPtr<SharedBuffer> finishEncoding() override;

    void encodeBytes(const String& key, std::span<const uint8_t>) override;
    void encodeBool(const String& key, bool) override;
    void encodeUInt32(const String& key, uint32_t) override;
    void encodeUInt64(const String& key, uint64_t) override;
    void encodeInt32(const String& key, int32_t) override;
    void encodeInt64(const String& key, int64_t) override;
    void encodeFloat(const String& key, float) override;
    void encodeDouble(const String& key, double) override;
    void encodeString(const String& key, const String&) override;

    void beginObject(const String& key) override;
    void endObject() override;

    void beginArray(const String& key) override;
    void beginArrayElement() override;
    void endArrayElement() override;
    void endArray() override;

    void encodeString(const String&);

    WTF::Persistence::Encoder m_encoder;
};

} // namespace WebCore

namespace WTF {
template<> struct EnumTraitsForPersistence<WebCore::KeyedEncoderGeneric::Type> {
    using values = EnumValues<
        WebCore::KeyedEncoderGeneric::Type,
        WebCore::KeyedEncoderGeneric::Type::Bytes,
        WebCore::KeyedEncoderGeneric::Type::Bool,
        WebCore::KeyedEncoderGeneric::Type::UInt32,
        WebCore::KeyedEncoderGeneric::Type::UInt64,
        WebCore::KeyedEncoderGeneric::Type::Int32,
        WebCore::KeyedEncoderGeneric::Type::Int64,
        WebCore::KeyedEncoderGeneric::Type::Float,
        WebCore::KeyedEncoderGeneric::Type::Double,
        WebCore::KeyedEncoderGeneric::Type::String,
        WebCore::KeyedEncoderGeneric::Type::BeginObject,
        WebCore::KeyedEncoderGeneric::Type::EndObject,
        WebCore::KeyedEncoderGeneric::Type::BeginArray,
        WebCore::KeyedEncoderGeneric::Type::BeginArrayElement,
        WebCore::KeyedEncoderGeneric::Type::EndArrayElement,
        WebCore::KeyedEncoderGeneric::Type::EndArray
    >;
};

} // namespace WTF
