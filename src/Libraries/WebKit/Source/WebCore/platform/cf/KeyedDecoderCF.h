/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 6, 2024.
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
#ifndef KeyedDecoderCF_h
#define KeyedDecoderCF_h

#include "KeyedCoding.h"
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class KeyedDecoderCF final : public KeyedDecoder {
    WTF_MAKE_TZONE_ALLOCATED(KeyedDecoderCF);
public:
    explicit KeyedDecoderCF(std::span<const uint8_t> data);
    ~KeyedDecoderCF() override;

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

    RetainPtr<CFDictionaryRef> m_rootDictionary;

    Vector<CFDictionaryRef, 16> m_dictionaryStack;
    Vector<CFArrayRef, 16> m_arrayStack;
    Vector<CFIndex> m_arrayIndexStack;
};

} // namespace WebCore

#endif // KeyedDecoderCF_h
