/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 31, 2024.
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
#include "KeyedEncoderCF.h"

#include "SharedBuffer.h"
#include <CoreFoundation/CoreFoundation.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(KeyedEncoderCF);

std::unique_ptr<KeyedEncoder> KeyedEncoder::encoder()
{
    return makeUnique<KeyedEncoderCF>();
}

static RetainPtr<CFMutableDictionaryRef> createDictionary()
{
    return adoptCF(CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks));
}

KeyedEncoderCF::KeyedEncoderCF()
    : m_rootDictionary(createDictionary())
{
    m_dictionaryStack.append(m_rootDictionary.get());
}
    
KeyedEncoderCF::~KeyedEncoderCF()
{
    ASSERT(m_dictionaryStack.size() == 1);
    ASSERT(m_dictionaryStack.last() == m_rootDictionary);
    ASSERT(m_arrayStack.isEmpty());
}

void KeyedEncoderCF::encodeBytes(const String& key, std::span<const uint8_t> bytes)
{
    RetainPtr data = adoptCF(CFDataCreateWithBytesNoCopy(kCFAllocatorDefault, bytes.data(), bytes.size(), kCFAllocatorNull));
    CFDictionarySetValue(m_dictionaryStack.last(), key.createCFString().get(), data.get());
}

void KeyedEncoderCF::encodeBool(const String& key, bool value)
{
    CFDictionarySetValue(m_dictionaryStack.last(), key.createCFString().get(), value ? kCFBooleanTrue : kCFBooleanFalse);
}

void KeyedEncoderCF::encodeUInt32(const String& key, uint32_t value)
{
    auto number = adoptCF(CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &value));
    CFDictionarySetValue(m_dictionaryStack.last(), key.createCFString().get(), number.get());
}
    
void KeyedEncoderCF::encodeUInt64(const String& key, uint64_t value)
{
    auto number = adoptCF(CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, &value));
    CFDictionarySetValue(m_dictionaryStack.last(), key.createCFString().get(), number.get());
}

void KeyedEncoderCF::encodeInt32(const String& key, int32_t value)
{
    auto number = adoptCF(CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &value));
    CFDictionarySetValue(m_dictionaryStack.last(), key.createCFString().get(), number.get());
}

void KeyedEncoderCF::encodeInt64(const String& key, int64_t value)
{
    auto number = adoptCF(CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, &value));
    CFDictionarySetValue(m_dictionaryStack.last(), key.createCFString().get(), number.get());
}

void KeyedEncoderCF::encodeFloat(const String& key, float value)
{
    auto number = adoptCF(CFNumberCreate(kCFAllocatorDefault, kCFNumberFloatType, &value));
    CFDictionarySetValue(m_dictionaryStack.last(), key.createCFString().get(), number.get());
}

void KeyedEncoderCF::encodeDouble(const String& key, double value)
{
    auto number = adoptCF(CFNumberCreate(kCFAllocatorDefault, kCFNumberDoubleType, &value));
    CFDictionarySetValue(m_dictionaryStack.last(), key.createCFString().get(), number.get());
}

void KeyedEncoderCF::encodeString(const String& key, const String& value)
{
    CFDictionarySetValue(m_dictionaryStack.last(), key.createCFString().get(), value.createCFString().get());
}

void KeyedEncoderCF::beginObject(const String& key)
{
    auto dictionary = createDictionary();
    CFDictionarySetValue(m_dictionaryStack.last(), key.createCFString().get(), dictionary.get());

    m_dictionaryStack.append(dictionary.get());
}

void KeyedEncoderCF::endObject()
{
    m_dictionaryStack.removeLast();
}

void KeyedEncoderCF::beginArray(const String& key)
{
    auto array = adoptCF(CFArrayCreateMutable(kCFAllocatorDefault, 0, &kCFTypeArrayCallBacks));
    CFDictionarySetValue(m_dictionaryStack.last(), key.createCFString().get(), array.get());

    m_arrayStack.append(array.get());
}

void KeyedEncoderCF::beginArrayElement()
{
    auto dictionary = createDictionary();
    CFArrayAppendValue(m_arrayStack.last(), dictionary.get());

    m_dictionaryStack.append(dictionary.get());
}

void KeyedEncoderCF::endArrayElement()
{
    m_dictionaryStack.removeLast();
}

void KeyedEncoderCF::endArray()
{
    m_arrayStack.removeLast();
}

RefPtr<SharedBuffer> KeyedEncoderCF::finishEncoding()
{
    auto data = adoptCF(CFPropertyListCreateData(kCFAllocatorDefault, m_rootDictionary.get(), kCFPropertyListBinaryFormat_v1_0, 0, nullptr));
    if (!data)
        return nullptr;
    return SharedBuffer::create(data.get());
}

} // namespace WebCore
