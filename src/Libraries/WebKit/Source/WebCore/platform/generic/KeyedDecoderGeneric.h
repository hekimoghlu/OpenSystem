/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 15, 2022.
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

namespace WebCore {

class KeyedDecoderGeneric final : public KeyedDecoder {
public:
    KeyedDecoderGeneric(std::span<const uint8_t> data);

    class Dictionary;
    using Array = Vector<std::unique_ptr<Dictionary>>;

private:
    WARN_UNUSED_RETURN bool decodeBytes(const String& key, std::span<const uint8_t>&) override;
    WARN_UNUSED_RETURN bool decodeBool(const String& key, bool&) override;
    WARN_UNUSED_RETURN bool decodeUInt32(const String& key, uint32_t&) override;
    WARN_UNUSED_RETURN bool decodeUInt64(const String& key, uint64_t&) override;
    WARN_UNUSED_RETURN bool decodeInt32(const String& key, int32_t&) override;
    WARN_UNUSED_RETURN bool decodeInt64(const String& key, int64_t&) override;
    WARN_UNUSED_RETURN bool decodeFloat(const String& key, float&) override;
    WARN_UNUSED_RETURN bool decodeDouble(const String& key, double&) override;
    WARN_UNUSED_RETURN bool decodeString(const String& key, String&) override;

    bool beginObject(const String& key) override;
    void endObject() override;

    bool beginArray(const String& key) override;
    bool beginArrayElement() override;
    void endArrayElement() override;
    void endArray() override;

    template<typename T>
    const T* getPointerFromDictionaryStack(const String& key);

    template<typename T> WARN_UNUSED_RETURN
    bool decodeSimpleValue(const String& key, T& result);

    std::unique_ptr<Dictionary> m_rootDictionary;
    Vector<Dictionary*, 16> m_dictionaryStack;
    Vector<Array*, 16> m_arrayStack;
    Vector<size_t, 16> m_arrayIndexStack;
};

} // namespace WebCore
