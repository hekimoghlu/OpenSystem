/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 2, 2023.
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

#include <wtf/text/UniquedStringImpl.h>

namespace WTF {

class RegisteredSymbolImpl;

// SymbolImpl is used to represent the symbol string impl.
// It is uniqued string impl, but is not registered in Atomic String tables, so it's not atomic.
class SUPPRESS_REFCOUNTED_WITHOUT_VIRTUAL_DESTRUCTOR SymbolImpl : public UniquedStringImpl {
public:
    using Flags = unsigned;
    static constexpr Flags s_flagDefault = 0u;
    static constexpr Flags s_flagIsNullSymbol = 0b001u;
    static constexpr Flags s_flagIsRegistered = 0b010u;
    static constexpr Flags s_flagIsPrivate = 0b100u;

    unsigned hashForSymbol() const { return m_hashForSymbolShiftedWithFlagCount >> s_flagCount; }
    bool isNullSymbol() const { return m_flags & s_flagIsNullSymbol; }
    bool isRegistered() const { return m_flags & s_flagIsRegistered; }
    bool isPrivate() const { return m_flags & s_flagIsPrivate; }

    SymbolRegistry* symbolRegistry() const;

    RegisteredSymbolImpl* asRegisteredSymbolImpl();

    WTF_EXPORT_PRIVATE static Ref<SymbolImpl> createNullSymbol();
    WTF_EXPORT_PRIVATE static Ref<SymbolImpl> create(StringImpl& rep);

    class StaticSymbolImpl final : private StringImplShape {
        WTF_MAKE_NONCOPYABLE(StaticSymbolImpl);
    public:
        template<unsigned characterCount>
        inline constexpr StaticSymbolImpl(const char (&characters)[characterCount], Flags = s_flagDefault);

        template<unsigned characterCount>
        inline constexpr StaticSymbolImpl(const char16_t (&characters)[characterCount], Flags = s_flagDefault);

        operator SymbolImpl&() { return *reinterpret_cast<SymbolImpl*>(this); }

        SUPPRESS_UNCOUNTED_MEMBER StringImpl* m_owner { nullptr }; // We do not make StaticSymbolImpl BufferSubstring. Thus we can make this nullptr.
        unsigned m_hashForSymbolShiftedWithFlagCount;
        Flags m_flags;
    };

protected:
    WTF_EXPORT_PRIVATE static unsigned nextHashForSymbol();

    friend class StringImpl;

    inline SymbolImpl(std::span<const LChar>, Ref<StringImpl>&&, Flags = s_flagDefault);
    inline SymbolImpl(std::span<const UChar>, Ref<StringImpl>&&, Flags = s_flagDefault);
    inline SymbolImpl(Flags = s_flagDefault);

    // The pointer to the owner string should be immediately following after the StringImpl layout,
    // since we would like to align the layout of SymbolImpl to the one of BufferSubstring StringImpl.
    SUPPRESS_UNCOUNTED_MEMBER StringImpl* m_owner;
    unsigned m_hashForSymbolShiftedWithFlagCount;
    Flags m_flags { s_flagDefault };
};
static_assert(sizeof(SymbolImpl) == sizeof(SymbolImpl::StaticSymbolImpl));

inline SymbolImpl::SymbolImpl(std::span<const LChar> characters, Ref<StringImpl>&& base, Flags flags)
    : UniquedStringImpl(CreateSymbol, characters)
    , m_owner(&base.leakRef())
    , m_hashForSymbolShiftedWithFlagCount(nextHashForSymbol())
    , m_flags(flags)
{
    static_assert(StringImpl::tailOffset<StringImpl*>() == OBJECT_OFFSETOF(SymbolImpl, m_owner));
}

inline SymbolImpl::SymbolImpl(std::span<const UChar> characters, Ref<StringImpl>&& base, Flags flags)
    : UniquedStringImpl(CreateSymbol, characters)
    , m_owner(&base.leakRef())
    , m_hashForSymbolShiftedWithFlagCount(nextHashForSymbol())
    , m_flags(flags)
{
    static_assert(StringImpl::tailOffset<StringImpl*>() == OBJECT_OFFSETOF(SymbolImpl, m_owner));
}

inline SymbolImpl::SymbolImpl(Flags flags)
    : UniquedStringImpl(CreateSymbol)
    , m_owner(StringImpl::empty())
    , m_hashForSymbolShiftedWithFlagCount(nextHashForSymbol())
    , m_flags(flags | s_flagIsNullSymbol)
{
    static_assert(StringImpl::tailOffset<StringImpl*>() == OBJECT_OFFSETOF(SymbolImpl, m_owner));
}

template<unsigned characterCount>
inline constexpr SymbolImpl::StaticSymbolImpl::StaticSymbolImpl(const char (&characters)[characterCount], Flags flags)
    : StringImplShape(s_refCountFlagIsStaticString, characterCount - 1, characters, s_hashFlag8BitBuffer | s_hashFlagDidReportCost | StringSymbol | BufferInternal | (StringHasher::computeLiteralHashAndMaskTop8Bits(characters) << s_flagCount), ConstructWithConstExpr)
    , m_hashForSymbolShiftedWithFlagCount(StringHasher::computeLiteralHashAndMaskTop8Bits(characters) << s_flagCount)
    , m_flags(flags)
{
}

template<unsigned characterCount>
inline constexpr SymbolImpl::StaticSymbolImpl::StaticSymbolImpl(const char16_t (&characters)[characterCount], Flags flags)
    : StringImplShape(s_refCountFlagIsStaticString, characterCount - 1, characters, s_hashFlagDidReportCost | StringSymbol | BufferInternal | (StringHasher::computeLiteralHashAndMaskTop8Bits(characters) << s_flagCount), ConstructWithConstExpr)
    , m_hashForSymbolShiftedWithFlagCount(StringHasher::computeLiteralHashAndMaskTop8Bits(characters) << s_flagCount)
    , m_flags(flags)
{
}

class PrivateSymbolImpl final : public SymbolImpl {
public:
    WTF_EXPORT_PRIVATE static Ref<PrivateSymbolImpl> create(StringImpl& rep);

private:
    PrivateSymbolImpl(std::span<const LChar> characters, Ref<StringImpl>&& base)
        : SymbolImpl(characters, WTFMove(base), s_flagIsPrivate)
    {
    }

    PrivateSymbolImpl(std::span<const UChar> characters, Ref<StringImpl>&& base)
        : SymbolImpl(characters, WTFMove(base), s_flagIsPrivate)
    {
    }
};

class RegisteredSymbolImpl final : public SymbolImpl {
private:
    friend class StringImpl;
    friend class SymbolImpl;
    friend class SymbolRegistry;

    SymbolRegistry* symbolRegistry() const { return m_symbolRegistry; }
    void clearSymbolRegistry() { m_symbolRegistry = nullptr; }

    static Ref<RegisteredSymbolImpl> create(StringImpl& rep, SymbolRegistry&);
    static Ref<RegisteredSymbolImpl> createPrivate(StringImpl& rep, SymbolRegistry&);

    RegisteredSymbolImpl(std::span<const LChar> characters, Ref<StringImpl>&& base, SymbolRegistry& registry, Flags flags = s_flagIsRegistered)
        : SymbolImpl(characters, WTFMove(base), flags)
        , m_symbolRegistry(&registry)
    {
    }

    RegisteredSymbolImpl(std::span<const UChar> characters, Ref<StringImpl>&& base, SymbolRegistry& registry, Flags flags = s_flagIsRegistered)
        : SymbolImpl(characters, WTFMove(base), flags)
        , m_symbolRegistry(&registry)
    {
    }

    SymbolRegistry* m_symbolRegistry;
};

inline unsigned StringImpl::symbolAwareHash() const
{
    if (isSymbol())
        return static_cast<const SymbolImpl*>(this)->hashForSymbol();
    return hash();
}

inline unsigned StringImpl::existingSymbolAwareHash() const
{
    if (isSymbol())
        return static_cast<const SymbolImpl*>(this)->hashForSymbol();
    return existingHash();
}

inline SymbolRegistry* SymbolImpl::symbolRegistry() const
{
    if (isRegistered())
        return static_cast<const RegisteredSymbolImpl*>(this)->symbolRegistry();
    return nullptr;
}

inline RegisteredSymbolImpl* SymbolImpl::asRegisteredSymbolImpl()
{
    ASSERT(isRegistered());
    return static_cast<RegisteredSymbolImpl*>(this);
}

#if ASSERT_ENABLED
// SymbolImpls created from StaticStringImpl will ASSERT
// in the generic ValueCheck<T>::checkConsistency
// as they are not allocated by fastMalloc.
// We don't currently have any way to detect that case
// so we ignore the consistency check for all SymbolImpls*.
template<> struct
ValueCheck<SymbolImpl*> {
    static void checkConsistency(const SymbolImpl*) { }
};

template<> struct
ValueCheck<const SymbolImpl*> {
    static void checkConsistency(const SymbolImpl*) { }
};
#endif // ASSERT_ENABLED

} // namespace WTF

using WTF::SymbolImpl;
using WTF::PrivateSymbolImpl;
using WTF::RegisteredSymbolImpl;
