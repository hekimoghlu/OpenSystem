/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 31, 2024.
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

#include <wtf/ArgumentCoder.h>
#include <wtf/HashFunctions.h>
#include <wtf/HashTraits.h>
#include <wtf/text/TextStream.h>
#include <wtf/text/WTFString.h>

namespace WebKit {

// MonotonicObjectIdentifier is similar to ObjectIdentifier but it can be monotonically
// increased in place.
// FIXME: merge ObjectIdentifier and MonotonicObjectIdentifier in one class.
template<typename T> class MonotonicObjectIdentifier {
public:
    MonotonicObjectIdentifier() = default;

    MonotonicObjectIdentifier(WTF::HashTableDeletedValueType)
        : m_identifier(hashTableDeletedValue())
    { }

    bool isHashTableDeletedValue() const { return m_identifier == hashTableDeletedValue(); }

    friend bool operator==(MonotonicObjectIdentifier, MonotonicObjectIdentifier) = default;

    bool operator>(const MonotonicObjectIdentifier& other) const
    {
        return m_identifier > other.m_identifier;
    }

    bool operator>=(const MonotonicObjectIdentifier& other) const
    {
        return m_identifier >= other.m_identifier;
    }

    bool operator<(const MonotonicObjectIdentifier& other) const
    {
        return m_identifier < other.m_identifier;
    }

    bool operator<=(const MonotonicObjectIdentifier& other) const
    {
        return m_identifier <= other.m_identifier;
    }

    MonotonicObjectIdentifier& increment()
    {
        ++m_identifier;
        return *this;
    }

    MonotonicObjectIdentifier next() const
    {
        return MonotonicObjectIdentifier(m_identifier + 1);
    }

    uint64_t toUInt64() const { return m_identifier; }
    explicit operator bool() const { return m_identifier; }

    String loggingString() const
    {
        return String::number(m_identifier);
    }

private:
    friend struct IPC::ArgumentCoder<MonotonicObjectIdentifier, void>;
    template<typename U> friend MonotonicObjectIdentifier<U> makeMonotonicObjectIdentifier(uint64_t);
    friend struct HashTraits<MonotonicObjectIdentifier>;
    template<typename U> friend struct MonotonicObjectIdentifierHash;

    static uint64_t hashTableDeletedValue() { return std::numeric_limits<uint64_t>::max(); }
    static bool isValidIdentifier(uint64_t identifier) { return identifier != hashTableDeletedValue(); }

    explicit MonotonicObjectIdentifier(uint64_t identifier)
        : m_identifier(identifier)
    {
    }

    uint64_t m_identifier { 0 };
};

template<typename T>
inline int64_t operator-(const MonotonicObjectIdentifier<T>& a, const MonotonicObjectIdentifier<T>& b)
{
    CheckedInt64 result = CheckedInt64(a.toUInt64() - b.toUInt64());
    return result.hasOverflowed() ? 0 : result.value();
}

template<typename T>
TextStream& operator<<(TextStream& ts, const MonotonicObjectIdentifier<T>& identifier)
{
    ts << identifier.toUInt64();
    return ts;
}

} // namespace WebKit
