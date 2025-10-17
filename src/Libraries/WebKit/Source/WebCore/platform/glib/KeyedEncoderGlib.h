/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 15, 2025.
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
#include <glib.h>
#include <wtf/Vector.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class KeyedEncoderGlib final : public KeyedEncoder {
public:
    KeyedEncoderGlib();
    ~KeyedEncoderGlib();

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

    GVariantBuilder m_variantBuilder;
    Vector<GVariantBuilder*, 16> m_variantBuilderStack;
    Vector<std::pair<String, GRefPtr<GVariantBuilder>>, 16> m_arrayStack;
    Vector<std::pair<String, GRefPtr<GVariantBuilder>>, 16> m_objectStack;
};

} // namespace WebCore
