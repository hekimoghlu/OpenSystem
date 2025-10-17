/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 26, 2023.
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
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class KeyedEncoderCF final : public KeyedEncoder {
    WTF_MAKE_TZONE_ALLOCATED(KeyedEncoderCF);
public:
    KeyedEncoderCF();
    ~KeyedEncoderCF();

private:
    RefPtr<WebCore::SharedBuffer> finishEncoding() final;

    void encodeBytes(const String& key, std::span<const uint8_t>) final;
    void encodeBool(const String& key, bool) final;
    void encodeUInt32(const String& key, uint32_t) final;
    void encodeUInt64(const String& key, uint64_t) final;
    void encodeInt32(const String& key, int32_t) final;
    void encodeInt64(const String& key, int64_t) final;
    void encodeFloat(const String& key, float) final;
    void encodeDouble(const String& key, double) final;
    void encodeString(const String& key, const String&) final;

    void beginObject(const String& key) final;
    void endObject() final;

    void beginArray(const String& key) final;
    void beginArrayElement() final;
    void endArrayElement() final;
    void endArray() final;

    RetainPtr<CFMutableDictionaryRef> m_rootDictionary;

    Vector<CFMutableDictionaryRef, 16> m_dictionaryStack;
    Vector<CFMutableArrayRef, 16> m_arrayStack;
};

} // namespace WebCore
