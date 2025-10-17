/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 17, 2024.
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
#include "KeyedEncoderGlib.h"

#include "SharedBuffer.h"
#include <wtf/glib/GSpanExtras.h>
#include <wtf/text/CString.h>

namespace WebCore {

std::unique_ptr<KeyedEncoder> KeyedEncoder::encoder()
{
    return makeUnique<KeyedEncoderGlib>();
}

KeyedEncoderGlib::KeyedEncoderGlib()
{
    g_variant_builder_init(&m_variantBuilder, G_VARIANT_TYPE("a{sv}"));
    m_variantBuilderStack.append(&m_variantBuilder);
}

KeyedEncoderGlib::~KeyedEncoderGlib()
{
    ASSERT(m_variantBuilderStack.size() == 1);
    ASSERT(m_variantBuilderStack.last() == &m_variantBuilder);
    ASSERT(m_arrayStack.isEmpty());
    ASSERT(m_objectStack.isEmpty());
}

void KeyedEncoderGlib::encodeBytes(const String& key, std::span<const uint8_t> bytes)
{
    GRefPtr<GBytes> gBytes = adoptGRef(g_bytes_new_static(bytes.data(), bytes.size()));
    g_variant_builder_add(m_variantBuilderStack.last(), "{sv}", key.utf8().data(), g_variant_new_from_bytes(G_VARIANT_TYPE("ay"), gBytes.get(), TRUE));
}

void KeyedEncoderGlib::encodeBool(const String& key, bool value)
{
    g_variant_builder_add(m_variantBuilderStack.last(), "{sv}", key.utf8().data(), g_variant_new_boolean(value));
}

void KeyedEncoderGlib::encodeUInt32(const String& key, uint32_t value)
{
    g_variant_builder_add(m_variantBuilderStack.last(), "{sv}", key.utf8().data(), g_variant_new_uint32(value));
}
    
void KeyedEncoderGlib::encodeUInt64(const String& key, uint64_t value)
{
    g_variant_builder_add(m_variantBuilderStack.last(), "{sv}", key.utf8().data(), g_variant_new_uint64(value));
}

void KeyedEncoderGlib::encodeInt32(const String& key, int32_t value)
{
    g_variant_builder_add(m_variantBuilderStack.last(), "{sv}", key.utf8().data(), g_variant_new_int32(value));
}

void KeyedEncoderGlib::encodeInt64(const String& key, int64_t value)
{
    g_variant_builder_add(m_variantBuilderStack.last(), "{sv}", key.utf8().data(), g_variant_new_int64(value));
}

void KeyedEncoderGlib::encodeFloat(const String& key, float value)
{
    encodeDouble(key, value);
}

void KeyedEncoderGlib::encodeDouble(const String& key, double value)
{
    g_variant_builder_add(m_variantBuilderStack.last(), "{sv}", key.utf8().data(), g_variant_new_double(value));
}

void KeyedEncoderGlib::encodeString(const String& key, const String& value)
{
    g_variant_builder_add(m_variantBuilderStack.last(), "{sv}", key.utf8().data(), g_variant_new_string(value.utf8().data()));
}

void KeyedEncoderGlib::beginObject(const String& key)
{
    GRefPtr<GVariantBuilder> builder = adoptGRef(g_variant_builder_new(G_VARIANT_TYPE("a{sv}")));
    m_objectStack.append(std::make_pair(key, builder));
    m_variantBuilderStack.append(builder.get());
}

void KeyedEncoderGlib::endObject()
{
    GVariantBuilder* builder = m_variantBuilderStack.takeLast();
    g_variant_builder_add(m_variantBuilderStack.last(), "{sv}", m_objectStack.last().first.utf8().data(), g_variant_builder_end(builder));
    m_objectStack.removeLast();
}

void KeyedEncoderGlib::beginArray(const String& key)
{
    m_arrayStack.append(std::make_pair(key, adoptGRef(g_variant_builder_new(G_VARIANT_TYPE("aa{sv}")))));
}

void KeyedEncoderGlib::beginArrayElement()
{
    m_variantBuilderStack.append(g_variant_builder_new(G_VARIANT_TYPE("a{sv}")));
}

void KeyedEncoderGlib::endArrayElement()
{
    GRefPtr<GVariantBuilder> variantBuilder = adoptGRef(m_variantBuilderStack.takeLast());
    g_variant_builder_add_value(m_arrayStack.last().second.get(), g_variant_builder_end(variantBuilder.get()));
}

void KeyedEncoderGlib::endArray()
{
    g_variant_builder_add(m_variantBuilderStack.last(), "{sv}", m_arrayStack.last().first.utf8().data(), g_variant_builder_end(m_arrayStack.last().second.get()));
    m_arrayStack.removeLast();
}

RefPtr<SharedBuffer> KeyedEncoderGlib::finishEncoding()
{
    g_assert(m_variantBuilderStack.last() == &m_variantBuilder);
    GRefPtr<GVariant> variant = g_variant_builder_end(&m_variantBuilder);
    GRefPtr<GBytes> data = adoptGRef(g_variant_get_data_as_bytes(variant.get()));
    return SharedBuffer::create(span(data));
}

} // namespace WebCore
