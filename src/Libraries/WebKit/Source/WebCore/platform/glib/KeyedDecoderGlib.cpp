/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 20, 2025.
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
#include "KeyedDecoderGlib.h"

#include <wtf/glib/GSpanExtras.h>
#include <wtf/text/CString.h>

namespace WebCore {

std::unique_ptr<KeyedDecoder> KeyedDecoder::decoder(std::span<const uint8_t> data)
{
    return makeUnique<KeyedDecoderGlib>(data);
}

KeyedDecoderGlib::KeyedDecoderGlib(std::span<const uint8_t> data)
{
    GRefPtr<GBytes> bytes = adoptGRef(g_bytes_new(data.data(), data.size()));
    GRefPtr<GVariant> variant = g_variant_new_from_bytes(G_VARIANT_TYPE("a{sv}"), bytes.get(), TRUE);
    m_dictionaryStack.append(dictionaryFromGVariant(variant.get()));
}

KeyedDecoderGlib::~KeyedDecoderGlib()
{
    ASSERT(m_dictionaryStack.size() == 1);
    ASSERT(m_arrayStack.isEmpty());
    ASSERT(m_arrayIndexStack.isEmpty());
}

UncheckedKeyHashMap<String, GRefPtr<GVariant>> KeyedDecoderGlib::dictionaryFromGVariant(GVariant* variant)
{
    UncheckedKeyHashMap<String, GRefPtr<GVariant>> dictionary;
    GVariantIter iter;
    g_variant_iter_init(&iter, variant);
    const char* key;
    GVariant* value;
    while (g_variant_iter_loop(&iter, "{&sv}", &key, &value)) {
        if (key)
            dictionary.set(String::fromUTF8(key), value);
    }
    return dictionary;
}

bool KeyedDecoderGlib::decodeBytes(const String& key, std::span<const uint8_t>& bytes)
{
    GRefPtr<GVariant> value = m_dictionaryStack.last().get(key);
    if (!value)
        return false;

    bytes = span(value);
    return true;
}

template<typename T, typename F>
bool KeyedDecoderGlib::decodeSimpleValue(const String& key, T& result, F getFunction)
{
    GRefPtr<GVariant> value = m_dictionaryStack.last().get(key);
    if (!value)
        return false;

    result = getFunction(value.get());
    return true;
}

bool KeyedDecoderGlib::decodeBool(const String& key, bool& result)
{
    return decodeSimpleValue(key, result, g_variant_get_boolean);
}

bool KeyedDecoderGlib::decodeUInt32(const String& key, uint32_t& result)
{
    return decodeSimpleValue(key, result, g_variant_get_uint32);
}

bool KeyedDecoderGlib::decodeUInt64(const String& key, uint64_t& result)
{
    return decodeSimpleValue(key, result, g_variant_get_uint64);
}

bool KeyedDecoderGlib::decodeInt32(const String& key, int32_t& result)
{
    return decodeSimpleValue(key, result, g_variant_get_int32);
}

bool KeyedDecoderGlib::decodeInt64(const String& key, int64_t& result)
{
    return decodeSimpleValue(key, result, g_variant_get_int64);
}

bool KeyedDecoderGlib::decodeFloat(const String& key, float& result)
{
    return decodeSimpleValue(key, result, g_variant_get_double);
}

bool KeyedDecoderGlib::decodeDouble(const String& key, double& result)
{
    return decodeSimpleValue(key, result, g_variant_get_double);
}

bool KeyedDecoderGlib::decodeString(const String& key, String& result)
{
    GRefPtr<GVariant> value = m_dictionaryStack.last().get(key);
    if (!value)
        return false;

    result = String::fromUTF8(g_variant_get_string(value.get(), nullptr));
    return true;
}

bool KeyedDecoderGlib::beginObject(const String& key)
{
    GRefPtr<GVariant> value = m_dictionaryStack.last().get(key);
    if (!value)
        return false;

    m_dictionaryStack.append(dictionaryFromGVariant(value.get()));
    return true;
}

void KeyedDecoderGlib::endObject()
{
    m_dictionaryStack.removeLast();
}

bool KeyedDecoderGlib::beginArray(const String& key)
{
    GRefPtr<GVariant> value = m_dictionaryStack.last().get(key);
    if (!value)
        return false;

    m_arrayStack.append(value.get());
    m_arrayIndexStack.append(0);
    return true;
}

bool KeyedDecoderGlib::beginArrayElement()
{
    if (m_arrayIndexStack.last() >= g_variant_n_children(m_arrayStack.last()))
        return false;

    GRefPtr<GVariant> variant = adoptGRef(g_variant_get_child_value(m_arrayStack.last(), m_arrayIndexStack.last()++));
    m_dictionaryStack.append(dictionaryFromGVariant(variant.get()));
    return true;
}

void KeyedDecoderGlib::endArrayElement()
{
    m_dictionaryStack.removeLast();
}

void KeyedDecoderGlib::endArray()
{
    m_arrayStack.removeLast();
    m_arrayIndexStack.removeLast();
}

} // namespace WebCore
