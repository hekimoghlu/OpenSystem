/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 19, 2024.
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

#include <wtf/Compiler.h>
#include <wtf/HashTraits.h>
#include <wtf/UUID.h>
#include <wtf/text/TextStream.h>
#include <wtf/text/WTFString.h>

namespace WTF {

class PrintStream;

template<typename RawValue>
struct ObjectIdentifierThreadSafeAccessTraits {
};

template<>
struct ObjectIdentifierThreadSafeAccessTraits<uint64_t> {
    WTF_EXPORT_PRIVATE static uint64_t generateIdentifierInternal();
};

template<>
struct ObjectIdentifierThreadSafeAccessTraits<UUID> {
    WTF_EXPORT_PRIVATE static UUID generateIdentifierInternal();
};

template<typename RawValue>
struct ObjectIdentifierMainThreadAccessTraits {
};

template<>
struct ObjectIdentifierMainThreadAccessTraits<uint64_t> {
    WTF_EXPORT_PRIVATE static uint64_t generateIdentifierInternal();
};

template<>
struct ObjectIdentifierMainThreadAccessTraits<UUID> {
    WTF_EXPORT_PRIVATE static UUID generateIdentifierInternal();
};

// Extracted from ObjectIdentifierGeneric to avoid binary bloat.
template<typename RawValue>
class ObjectIdentifierGenericBase {
};

template<>
class ObjectIdentifierGenericBase<uint64_t> {
public:
    using RawValue = uint64_t;

    bool isHashTableDeletedValue() const { return m_identifier == hashTableDeletedValue(); }

    RawValue toUInt64() const { return toRawValue(); } // Use `toRawValue` instead.
    RawValue toRawValue() const { return m_identifier; }

    String loggingString() const
    {
        return String::number(m_identifier);
    }

    static constexpr bool isValidIdentifier(RawValue identifier) { return identifier && identifier != hashTableDeletedValue(); }

protected:
    explicit constexpr ObjectIdentifierGenericBase(RawValue identifier)
        : m_identifier(identifier)
    {
    }

    ObjectIdentifierGenericBase() = default;
    ~ObjectIdentifierGenericBase() = default;
    ObjectIdentifierGenericBase(HashTableDeletedValueType) : m_identifier(hashTableDeletedValue()) { }

    static constexpr RawValue hashTableDeletedValue() { return std::numeric_limits<RawValue>::max(); }

private:
    RawValue m_identifier { 0 };
};

template<>
class ObjectIdentifierGenericBase<UUID> {
public:
    using RawValue = UUID;

    bool isHashTableDeletedValue() const { return m_identifier == hashTableDeletedValue(); }

    RawValue toRawValue() const { return m_identifier; }

    String loggingString() const
    {
        return m_identifier.toString();
    }

    static constexpr bool isValidIdentifier(RawValue identifier) { return identifier && !identifier.isHashTableDeletedValue(); }

protected:
    explicit constexpr ObjectIdentifierGenericBase(RawValue identifier)
        : m_identifier(identifier)
    {
    }

    ObjectIdentifierGenericBase() = default;
    ~ObjectIdentifierGenericBase() = default;
    ObjectIdentifierGenericBase(HashTableDeletedValueType) : m_identifier(hashTableDeletedValue()) { }

    static RawValue hashTableDeletedValue() { return UUID { HashTableDeletedValue }; }

private:
    RawValue m_identifier { UUID::MarkableTraits::emptyValue() };
};

template<typename T, typename ThreadSafety, typename RawValue>
class ObjectIdentifierGeneric : public ObjectIdentifierGenericBase<RawValue> {
public:
    static ObjectIdentifierGeneric generate()
    {
        RELEASE_ASSERT(!m_generationProtected);
        return ObjectIdentifierGeneric { ThreadSafety::generateIdentifierInternal(), AssumeValidIdValue };
    }

    static void enableGenerationProtection()
    {
        m_generationProtected = true;
    }

    explicit constexpr ObjectIdentifierGeneric(RawValue identifier)
        : ObjectIdentifierGenericBase<RawValue>(identifier)
    {
        RELEASE_ASSERT(ObjectIdentifierGenericBase<RawValue>::isValidIdentifier(identifier));
    }

    bool isHashTableEmptyValue() const { return !ObjectIdentifierGenericBase<RawValue>::toRawValue(); }

    // Do not call this constructor explicitly, it should only be used by the Hashtable implementation.
    ObjectIdentifierGeneric(HashTableDeletedValueType) : ObjectIdentifierGenericBase<RawValue>(HashTableDeletedValue) { }

    struct MarkableTraits {
        static bool isEmptyValue(ObjectIdentifierGeneric identifier) { return !identifier.toRawValue(); }
        static constexpr ObjectIdentifierGeneric emptyValue() { return ObjectIdentifierGeneric(InvalidIdValue); }
    };

private:
    friend struct HashTraits<ObjectIdentifierGeneric>;
    template<typename U, typename V> friend struct ObjectIdentifierGenericHash;

    enum AssumeValidId { AssumeValidIdValue };
    explicit constexpr ObjectIdentifierGeneric(RawValue identifier, AssumeValidId)
        : ObjectIdentifierGenericBase<RawValue>(identifier)
    {
        ASSERT(!!identifier);
    }

    enum InvalidId { InvalidIdValue };
    ObjectIdentifierGeneric(InvalidId)
    {
    }

    inline static bool m_generationProtected { false };
};

template<typename T, typename RawValue> using ObjectIdentifier = ObjectIdentifierGeneric<T, ObjectIdentifierMainThreadAccessTraits<RawValue>, RawValue>;
template<typename T, typename RawValue> using AtomicObjectIdentifier = ObjectIdentifierGeneric<T, ObjectIdentifierThreadSafeAccessTraits<RawValue>, RawValue>;

inline void add(Hasher& hasher, const ObjectIdentifierGenericBase<uint64_t>& identifier)
{
    add(hasher, identifier.toRawValue());
}

inline void add(Hasher& hasher, const ObjectIdentifierGenericBase<UUID>& identifier)
{
    add(hasher, identifier.toRawValue());
}

template<typename RawValue>
struct ObjectIdentifierGenericBaseHash {
};

template<>
struct ObjectIdentifierGenericBaseHash<uint64_t> {
    static unsigned hash(const ObjectIdentifierGenericBase<uint64_t>& identifier) { return intHash(identifier.toUInt64()); }
    static bool equal(const ObjectIdentifierGenericBase<uint64_t>& a, const ObjectIdentifierGenericBase<uint64_t>& b) { return a.toUInt64() == b.toUInt64(); }
    static constexpr bool safeToCompareToEmptyOrDeleted = true;
};

template<>
struct ObjectIdentifierGenericBaseHash<UUID> {
    static unsigned hash(const ObjectIdentifierGenericBase<UUID>& identifier) { return UUIDHash::hash(identifier.toRawValue()); }
    static bool equal(const ObjectIdentifierGenericBase<UUID>& a, const ObjectIdentifierGenericBase<UUID>& b) { return UUIDHash::equal(a.toRawValue(), b.toRawValue()); }
    static constexpr bool safeToCompareToEmptyOrDeleted = true;
};

template<typename T, typename U, typename V> struct HashTraits<ObjectIdentifierGeneric<T, U, V>> : SimpleClassHashTraits<ObjectIdentifierGeneric<T, U, V>> {
    using ValueType = ObjectIdentifierGeneric<T, U, V>;
    using PeekType = std::optional<ValueType>;
    using TakeType = std::optional<ValueType>;

    static ValueType emptyValue() { return ValueType { ValueType::InvalidIdValue }; }
    static bool isEmptyValue(ValueType value) { return value.isHashTableEmptyValue(); }

    static PeekType peek(ValueType identifier)
    {
        if (isEmptyValue(identifier))
            return std::nullopt;
        return identifier;
    }

    static TakeType take(ValueType identifier)
    {
        if (isEmptyValue(identifier))
            return std::nullopt;
        return identifier;
    }
};

template<typename T, typename U, typename V> struct DefaultHash<ObjectIdentifierGeneric<T, U, V>> : ObjectIdentifierGenericBaseHash<V> { };

WTF_EXPORT_PRIVATE TextStream& operator<<(TextStream&, const ObjectIdentifierGenericBase<uint64_t>&);

WTF_EXPORT_PRIVATE TextStream& operator<<(TextStream&, const ObjectIdentifierGenericBase<UUID>&);

WTF_EXPORT_PRIVATE void printInternal(PrintStream&, const ObjectIdentifierGenericBase<uint64_t>&);

WTF_EXPORT_PRIVATE void printInternal(PrintStream&, const ObjectIdentifierGenericBase<UUID>&);

template<typename RawValue>
class ObjectIdentifierGenericBaseStringTypeAdapter {
};

template<>
class ObjectIdentifierGenericBaseStringTypeAdapter<uint64_t> {
public:
    unsigned length() const { return lengthOfIntegerAsString(m_identifier); }
    bool is8Bit() const { return true; }
    template<typename CharacterType> void writeTo(std::span<CharacterType> destination) const { writeIntegerToBuffer(m_identifier, destination); }
protected:
    explicit ObjectIdentifierGenericBaseStringTypeAdapter(uint64_t identifier)
        : m_identifier(identifier) { }
private:
    uint64_t m_identifier;
};

template<typename T, typename ThreadSafety>
class StringTypeAdapter<ObjectIdentifierGeneric<T, ThreadSafety, uint64_t>> : public ObjectIdentifierGenericBaseStringTypeAdapter<uint64_t> {
public:
    explicit StringTypeAdapter(ObjectIdentifierGeneric<T, ThreadSafety, uint64_t> identifier)
        : ObjectIdentifierGenericBaseStringTypeAdapter(identifier.toRawValue()) { }
};

template<typename T, typename ThreadSafety>
class StringTypeAdapter<ObjectIdentifierGeneric<T, ThreadSafety, UUID>> : public StringTypeAdapter<UUID> {
public:
    explicit StringTypeAdapter(ObjectIdentifierGeneric<T, ThreadSafety, UUID> identifier)
        : StringTypeAdapter<UUID>(identifier.toRawValue()) { }
};

template<typename T, typename ThreadSafety>
bool operator==(const ObjectIdentifierGeneric<T, ThreadSafety, UUID>& a, const ObjectIdentifierGeneric<T, ThreadSafety, UUID>& b)
{
    return a.toRawValue() == b.toRawValue();
}

template<typename T, typename ThreadSafety>
bool operator==(const ObjectIdentifierGeneric<T, ThreadSafety, uint64_t>& a, const ObjectIdentifierGeneric<T, ThreadSafety, uint64_t>& b)
{
    return a.toRawValue() == b.toRawValue();
}

template<typename T, typename ThreadSafety>
bool operator>(const ObjectIdentifierGeneric<T, ThreadSafety, uint64_t>& a, const ObjectIdentifierGeneric<T, ThreadSafety, uint64_t>& b)
{
    return a.toRawValue() > b.toRawValue();
}

template<typename T, typename ThreadSafety>
bool operator>=(const ObjectIdentifierGeneric<T, ThreadSafety, uint64_t>& a, const ObjectIdentifierGeneric<T, ThreadSafety, uint64_t>& b)
{
    return a.toRawValue() >= b.toRawValue();
}

template<typename T, typename ThreadSafety>
bool operator<(const ObjectIdentifierGeneric<T, ThreadSafety, uint64_t>& a, const ObjectIdentifierGeneric<T, ThreadSafety, uint64_t>& b)
{
    return a.toRawValue() < b.toRawValue();
}

template<typename T, typename ThreadSafety>
bool operator<=(const ObjectIdentifierGeneric<T, ThreadSafety, uint64_t>& a, const ObjectIdentifierGeneric<T, ThreadSafety, uint64_t>& b)
{
    return a.toRawValue() <= b.toRawValue();
}

} // namespace WTF

using WTF::AtomicObjectIdentifier;
using WTF::ObjectIdentifierGenericBase;
using WTF::ObjectIdentifierGeneric;
using WTF::ObjectIdentifier;
