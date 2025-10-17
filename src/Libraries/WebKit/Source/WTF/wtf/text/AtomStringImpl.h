/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 13, 2025.
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

#include <wtf/TypeCasts.h>
#include <wtf/text/UniquedStringImpl.h>

namespace WTF {

class AtomStringTable;

class SUPPRESS_REFCOUNTED_WITHOUT_VIRTUAL_DESTRUCTOR AtomStringImpl final : public UniquedStringImpl {
public:
    WTF_EXPORT_PRIVATE static RefPtr<AtomStringImpl> lookUp(std::span<const LChar>);
    WTF_EXPORT_PRIVATE static RefPtr<AtomStringImpl> lookUp(std::span<const UChar>);
    static RefPtr<AtomStringImpl> lookUp(StringImpl*);

    static void remove(AtomStringImpl*);

    WTF_EXPORT_PRIVATE static RefPtr<AtomStringImpl> add(std::span<const LChar>);
    WTF_EXPORT_PRIVATE static RefPtr<AtomStringImpl> add(std::span<const UChar>);
    ALWAYS_INLINE static RefPtr<AtomStringImpl> add(std::span<const char> characters);

    WTF_EXPORT_PRIVATE static RefPtr<AtomStringImpl> add(HashTranslatorCharBuffer<LChar>&);
    WTF_EXPORT_PRIVATE static RefPtr<AtomStringImpl> add(HashTranslatorCharBuffer<UChar>&);

    WTF_EXPORT_PRIVATE static RefPtr<AtomStringImpl> add(StringImpl*, unsigned offset, unsigned length);
    ALWAYS_INLINE static RefPtr<AtomStringImpl> add(StringImpl*);
    ALWAYS_INLINE static RefPtr<AtomStringImpl> add(RefPtr<StringImpl>&&);
    WTF_EXPORT_PRIVATE static RefPtr<AtomStringImpl> add(const StaticStringImpl*);
    ALWAYS_INLINE static Ref<AtomStringImpl> add(ASCIILiteral);

    // Not using the add() naming to encourage developers to call add(ASCIILiteral) when they have a string literal.
    ALWAYS_INLINE static RefPtr<AtomStringImpl> addCString(const char*);

    // Returns null if the input data contains an invalid UTF-8 sequence.
    static RefPtr<AtomStringImpl> add(std::span<const char8_t>);

#if USE(CF)
    WTF_EXPORT_PRIVATE static RefPtr<AtomStringImpl> add(CFStringRef);
#endif

    template<typename StringTableProvider>
    ALWAYS_INLINE static RefPtr<AtomStringImpl> addWithStringTableProvider(StringTableProvider&, StringImpl*);

#if ASSERT_ENABLED
    WTF_EXPORT_PRIVATE static bool isInAtomStringTable(StringImpl*);
#endif

private:
    AtomStringImpl() = delete;

    ALWAYS_INLINE static Ref<AtomStringImpl> add(StringImpl&);
    ALWAYS_INLINE static Ref<AtomStringImpl> add(Ref<StringImpl>&&);
    WTF_EXPORT_PRIVATE static Ref<AtomStringImpl> addLiteral(std::span<const LChar>);

    ALWAYS_INLINE static Ref<AtomStringImpl> add(AtomStringTable&, StringImpl&);

    WTF_EXPORT_PRIVATE static Ref<AtomStringImpl> addSlowCase(StringImpl&);
    WTF_EXPORT_PRIVATE static Ref<AtomStringImpl> addSlowCase(Ref<StringImpl>&&);
    WTF_EXPORT_PRIVATE static Ref<AtomStringImpl> addSlowCase(AtomStringTable&, StringImpl&);

    WTF_EXPORT_PRIVATE static RefPtr<AtomStringImpl> lookUpSlowCase(StringImpl&);
};

inline RefPtr<AtomStringImpl> AtomStringImpl::lookUp(StringImpl* string)
{
    if (!string)
        return nullptr;
    if (auto* atom = dynamicDowncast<AtomStringImpl>(*string))
        return atom;
    return lookUpSlowCase(*string);
}

ALWAYS_INLINE RefPtr<AtomStringImpl> AtomStringImpl::add(std::span<const char> characters)
{
    return add(byteCast<LChar>(characters));
}

ALWAYS_INLINE RefPtr<AtomStringImpl> AtomStringImpl::add(StringImpl* string)
{
    if (!string)
        return nullptr;
    return add(*string);
}

ALWAYS_INLINE RefPtr<AtomStringImpl> AtomStringImpl::add(RefPtr<StringImpl>&& string)
{
    if (!string)
        return nullptr;
    return add(string.releaseNonNull());
}

ALWAYS_INLINE Ref<AtomStringImpl> AtomStringImpl::add(ASCIILiteral literal)
{
    return addLiteral(literal.span8());
}

ALWAYS_INLINE RefPtr<AtomStringImpl> AtomStringImpl::addCString(const char* s)
{
    return s ? add(unsafeSpan8(s)) : nullptr;
}

template<typename StringTableProvider>
ALWAYS_INLINE RefPtr<AtomStringImpl> AtomStringImpl::addWithStringTableProvider(StringTableProvider& stringTableProvider, StringImpl* string)
{
    if (!string)
        return nullptr;
    return add(*stringTableProvider.atomStringTable(), *string);
}

ALWAYS_INLINE Ref<AtomStringImpl> AtomStringImpl::add(StringImpl& string)
{
    if (auto* atom = dynamicDowncast<AtomStringImpl>(string)) {
        ASSERT_WITH_MESSAGE(!string.length() || isInAtomStringTable(&string), "The atom string comes from an other thread!");
        return *atom;
    }
    return addSlowCase(string);
}

ALWAYS_INLINE Ref<AtomStringImpl> AtomStringImpl::add(Ref<StringImpl>&& string)
{
    if (string->isAtom()) {
        ASSERT_WITH_MESSAGE(!string->length() || isInAtomStringTable(string.ptr()), "The atom string comes from an other thread!");
        return static_reference_cast<AtomStringImpl>(WTFMove(string));
    }
    return addSlowCase(WTFMove(string));
}

ALWAYS_INLINE Ref<AtomStringImpl> AtomStringImpl::add(AtomStringTable& stringTable, StringImpl& string)
{
    if (auto* atom = dynamicDowncast<AtomStringImpl>(string)) {
        ASSERT_WITH_MESSAGE(!string.length() || isInAtomStringTable(&string), "The atom string comes from an other thread!");
        return *atom;
    }
    return addSlowCase(stringTable, string);
}

#if ASSERT_ENABLED

// AtomStringImpls created from StaticStringImpl will ASSERT in the generic ValueCheck<T>::checkConsistency,
// as they are not allocated by fastMalloc. We don't currently have any way to detect that case, so we don't
// do any consistency check for AtomStringImpl*.

template<> struct ValueCheck<AtomStringImpl*> {
    static void checkConsistency(const AtomStringImpl*) { }
};

template<> struct ValueCheck<const AtomStringImpl*> {
    static void checkConsistency(const AtomStringImpl*) { }
};

#endif // ASSERT_ENABLED

} // namespace WTF

SPECIALIZE_TYPE_TRAITS_BEGIN(WTF::AtomStringImpl) \
    static bool isType(const WTF::StringImpl& string) { return string.isAtom(); } \
SPECIALIZE_TYPE_TRAITS_END()

using WTF::AtomStringImpl;
