/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 12, 2022.
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
#include <wtf/text/CString.h>

#include <string.h>
#include <wtf/CheckedArithmetic.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/StdLibExtras.h>
#include <wtf/text/StringCommon.h>
#include <wtf/text/SuperFastHash.h>

namespace WTF {

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(CStringBuffer);

Ref<CStringBuffer> CStringBuffer::createUninitialized(size_t length)
{
    // The +1 is for the terminating null character.
    size_t size = Checked<size_t>(sizeof(CStringBuffer)) + length + 1U;
    auto* stringBuffer = static_cast<CStringBuffer*>(CStringBufferMalloc::malloc(size));

    Ref buffer = adoptRef(*new (NotNull, stringBuffer) CStringBuffer(length));
    buffer->mutableSpanIncludingNullTerminator()[length] = '\0';
    return buffer;
}

CString::CString(const char* string)
{
    if (!string)
        return;

    init(unsafeSpan(string));
}

CString::CString(std::span<const char> string)
{
    if (!string.data()) {
        ASSERT(string.empty());
        return;
    }

    init(string);
}

void CString::init(std::span<const char> string)
{
    ASSERT(string.data());

    m_buffer = CStringBuffer::createUninitialized(string.size());
    memcpySpan(m_buffer->mutableSpan(), string);
}

std::span<char> CString::mutableSpan()
{
    copyBufferIfNeeded();
    if (!m_buffer)
        return { };
    return m_buffer->mutableSpan();
}

std::span<char> CString::mutableSpanIncludingNullTerminator()
{
    copyBufferIfNeeded();
    if (!m_buffer)
        return { };
    return m_buffer->mutableSpanIncludingNullTerminator();
}

CString CString::newUninitialized(size_t length, std::span<char>& characterBuffer)
{
    CString result;
    result.m_buffer = CStringBuffer::createUninitialized(length);
    characterBuffer = result.m_buffer->mutableSpan();
    return result;
}

void CString::copyBufferIfNeeded()
{
    if (!m_buffer || m_buffer->hasOneRef())
        return;

    RefPtr<CStringBuffer> buffer = WTFMove(m_buffer);
    size_t length = buffer->length();
    m_buffer = CStringBuffer::createUninitialized(length);
    memcpySpan(m_buffer->mutableSpanIncludingNullTerminator(), buffer->unsafeSpanIncludingNullTerminator());
}

bool CString::isSafeToSendToAnotherThread() const
{
    return !m_buffer || m_buffer->hasOneRef();
}

void CString::grow(size_t newLength)
{
    ASSERT(newLength > length());

    auto newBuffer = CStringBuffer::createUninitialized(newLength);
    memcpySpan(newBuffer->mutableSpanIncludingNullTerminator(), m_buffer->unsafeSpanIncludingNullTerminator());
    m_buffer = WTFMove(newBuffer);
}

bool operator==(const CString& a, const CString& b)
{
    if (a.isNull() != b.isNull())
        return false;
    if (a.length() != b.length())
        return false;
    return equal(a.span().data(), b.span());
}

unsigned CString::hash() const
{
    if (isNull())
        return 0;
    SuperFastHash hasher;
    for (auto character : span())
        hasher.addCharacter(character);
    return hasher.hash();
}

bool operator<(const CString& a, const CString& b)
{
    if (a.isNull())
        return !b.isNull();
    if (b.isNull())
        return false;
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    return strcmp(a.data(), b.data()) < 0;
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
}

bool CStringHash::equal(const CString& a, const CString& b)
{
    if (a.isHashTableDeletedValue())
        return b.isHashTableDeletedValue();
    if (b.isHashTableDeletedValue())
        return false;
    return a == b;
}

} // namespace WTF
