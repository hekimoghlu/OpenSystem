/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 2, 2024.
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
#include "KeyedDecoderCF.h"

#include <wtf/TZoneMallocInlines.h>
#include <wtf/cf/TypeCastsCF.h>
#include <wtf/cf/VectorCF.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(KeyedDecoderCF);

std::unique_ptr<KeyedDecoder> KeyedDecoder::decoder(std::span<const uint8_t> data)
{
    return makeUnique<KeyedDecoderCF>(data);
}

KeyedDecoderCF::KeyedDecoderCF(std::span<const uint8_t> data)
{
    auto cfData = adoptCF(CFDataCreateWithBytesNoCopy(kCFAllocatorDefault, data.data(), data.size(), kCFAllocatorNull));
    auto cfPropertyList = adoptCF(CFPropertyListCreateWithData(kCFAllocatorDefault, cfData.get(), kCFPropertyListImmutable, nullptr, nullptr));

    if (dynamic_cf_cast<CFDictionaryRef>(cfPropertyList.get()))
        m_rootDictionary = adoptCF(static_cast<CFDictionaryRef>(cfPropertyList.leakRef()));
    else
        m_rootDictionary = adoptCF(CFDictionaryCreate(kCFAllocatorDefault, nullptr, nullptr, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks));
    m_dictionaryStack.append(m_rootDictionary.get());
}

KeyedDecoderCF::~KeyedDecoderCF()
{
    ASSERT(m_dictionaryStack.size() == 1);
    ASSERT(m_dictionaryStack.last() == m_rootDictionary);
    ASSERT(m_arrayStack.isEmpty());
    ASSERT(m_arrayIndexStack.isEmpty());
}

bool KeyedDecoderCF::decodeBytes(const String& key, std::span<const uint8_t>& bytes)
{
    auto data = dynamic_cf_cast<CFDataRef>(CFDictionaryGetValue(m_dictionaryStack.last(), key.createCFString().get()));
    if (!data)
        return false;

    bytes = span(data);
    return true;
}

bool KeyedDecoderCF::decodeBool(const String& key, bool& result)
{
    auto boolean = dynamic_cf_cast<CFBooleanRef>(CFDictionaryGetValue(m_dictionaryStack.last(), key.createCFString().get()));
    if (!boolean)
        return false;

    result = CFBooleanGetValue(boolean);
    return true;
}

bool KeyedDecoderCF::decodeUInt32(const String& key, uint32_t& result)
{
    return decodeInt32(key, reinterpret_cast<int32_t&>(result));
}
    
bool KeyedDecoderCF::decodeUInt64(const String& key, uint64_t& result)
{
    return decodeInt64(key, reinterpret_cast<int64_t&>(result));
}

bool KeyedDecoderCF::decodeInt32(const String& key, int32_t& result)
{
    auto number = dynamic_cf_cast<CFNumberRef>(CFDictionaryGetValue(m_dictionaryStack.last(), key.createCFString().get()));
    if (!number)
        return false;

    return CFNumberGetValue(number, kCFNumberSInt32Type, &result);
}

bool KeyedDecoderCF::decodeInt64(const String& key, int64_t& result)
{
    auto number = dynamic_cf_cast<CFNumberRef>(CFDictionaryGetValue(m_dictionaryStack.last(), key.createCFString().get()));
    if (!number)
        return false;

    return CFNumberGetValue(number, kCFNumberSInt64Type, &result);
}

bool KeyedDecoderCF::decodeFloat(const String& key, float& result)
{
    auto number = dynamic_cf_cast<CFNumberRef>(CFDictionaryGetValue(m_dictionaryStack.last(), key.createCFString().get()));
    if (!number)
        return false;

    return CFNumberGetValue(number, kCFNumberFloatType, &result);
}

bool KeyedDecoderCF::decodeDouble(const String& key, double& result)
{
    auto number = dynamic_cf_cast<CFNumberRef>(CFDictionaryGetValue(m_dictionaryStack.last(), key.createCFString().get()));
    if (!number)
        return false;

    return CFNumberGetValue(number, kCFNumberDoubleType, &result);
}

bool KeyedDecoderCF::decodeString(const String& key, String& result)
{
    auto string = dynamic_cf_cast<CFStringRef>(CFDictionaryGetValue(m_dictionaryStack.last(), key.createCFString().get()));
    if (!string)
        return false;

    result = string;
    return true;
}

bool KeyedDecoderCF::beginObject(const String& key)
{
    auto dictionary = dynamic_cf_cast<CFDictionaryRef>(CFDictionaryGetValue(m_dictionaryStack.last(), key.createCFString().get()));
    if (!dictionary)
        return false;

    m_dictionaryStack.append(dictionary);
    return true;
}

void KeyedDecoderCF::endObject()
{
    m_dictionaryStack.removeLast();
}

bool KeyedDecoderCF::beginArray(const String& key)
{
    auto array = dynamic_cf_cast<CFArrayRef>(CFDictionaryGetValue(m_dictionaryStack.last(), key.createCFString().get()));
    if (!array)
        return false;

    for (CFIndex i = 0; i < CFArrayGetCount(array); ++i) {
        CFTypeRef object = CFArrayGetValueAtIndex(array, i);
        if (CFGetTypeID(object) != CFDictionaryGetTypeID())
            return false;
    }

    m_arrayStack.append(array);
    m_arrayIndexStack.append(0);
    return true;
}

bool KeyedDecoderCF::beginArrayElement()
{
    if (m_arrayIndexStack.last() >= CFArrayGetCount(m_arrayStack.last()))
        return false;

    auto dictionary = checked_cf_cast<CFDictionaryRef>(CFArrayGetValueAtIndex(m_arrayStack.last(), m_arrayIndexStack.last()++));
    m_dictionaryStack.append(dictionary);
    return true;
}

void KeyedDecoderCF::endArrayElement()
{
    m_dictionaryStack.removeLast();
}

void KeyedDecoderCF::endArray()
{
    m_arrayStack.removeLast();
    m_arrayIndexStack.removeLast();
}

} // namespace WebCore
