/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 23, 2023.
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
#include "BuiltinNames.h"

#include "IdentifierInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC {
namespace Symbols {

#define INITIALIZE_BUILTIN_STATIC_SYMBOLS(name) SymbolImpl::StaticSymbolImpl name##Symbol { "Symbol." #name };
JSC_COMMON_PRIVATE_IDENTIFIERS_EACH_WELL_KNOWN_SYMBOL(INITIALIZE_BUILTIN_STATIC_SYMBOLS)
#undef INITIALIZE_BUILTIN_STATIC_SYMBOLS

SymbolImpl::StaticSymbolImpl intlLegacyConstructedSymbol { "IntlLegacyConstructedSymbol" };

#define INITIALIZE_BUILTIN_PRIVATE_NAMES(name) SymbolImpl::StaticSymbolImpl name##PrivateName { #name, SymbolImpl::s_flagIsPrivate };
JSC_FOREACH_BUILTIN_FUNCTION_NAME(INITIALIZE_BUILTIN_PRIVATE_NAMES)
JSC_COMMON_PRIVATE_IDENTIFIERS_EACH_PROPERTY_NAME(INITIALIZE_BUILTIN_PRIVATE_NAMES)
#undef INITIALIZE_BUILTIN_PRIVATE_NAMES

SymbolImpl::StaticSymbolImpl dollarVMPrivateName { "$vm", SymbolImpl::s_flagIsPrivate };
SymbolImpl::StaticSymbolImpl polyProtoPrivateName { "PolyProto", SymbolImpl::s_flagIsPrivate };

} // namespace Symbols

#define INITIALIZE_BUILTIN_NAMES_IN_JSC(name) , m_##name(JSC::Identifier::fromString(vm, #name ""_s))
#define INITIALIZE_BUILTIN_SYMBOLS_IN_JSC(name) \
    , m_##name##Symbol(JSC::Identifier::fromUid(vm, &static_cast<SymbolImpl&>(JSC::Symbols::name##Symbol))) \
    , m_##name##SymbolPrivateIdentifier(JSC::Identifier::fromString(vm, #name ""_s))

#define INITIALIZE_PUBLIC_TO_PRIVATE_ENTRY(name) \
    do { \
        SymbolImpl* symbol = &static_cast<SymbolImpl&>(JSC::Symbols::name##PrivateName); \
        checkPublicToPrivateMapConsistency(symbol); \
        m_privateNameSet.add(symbol); \
    } while (0);

#define INITIALIZE_WELL_KNOWN_SYMBOL_PUBLIC_TO_PRIVATE_ENTRY(name) \
    do { \
        SymbolImpl* symbol = static_cast<SymbolImpl*>(m_##name##Symbol.impl()); \
        m_wellKnownSymbolsMap.add(m_##name##SymbolPrivateIdentifier.impl(), symbol); \
    } while (0);

WTF_MAKE_TZONE_ALLOCATED_IMPL(BuiltinNames);

// We treat the dollarVM name as a special case below for $vm (because CommonIdentifiers does not
// yet support the $ character).
BuiltinNames::BuiltinNames(VM& vm, CommonIdentifiers* commonIdentifiers)
    : m_emptyIdentifier(commonIdentifiers->emptyIdentifier)
    JSC_FOREACH_BUILTIN_FUNCTION_NAME(INITIALIZE_BUILTIN_NAMES_IN_JSC)
    JSC_COMMON_PRIVATE_IDENTIFIERS_EACH_PROPERTY_NAME(INITIALIZE_BUILTIN_NAMES_IN_JSC)
    JSC_COMMON_PRIVATE_IDENTIFIERS_EACH_WELL_KNOWN_SYMBOL(INITIALIZE_BUILTIN_SYMBOLS_IN_JSC)
    , m_intlLegacyConstructedSymbol(JSC::Identifier::fromUid(vm, &static_cast<SymbolImpl&>(Symbols::intlLegacyConstructedSymbol)))
    , m_dollarVMName(Identifier::fromString(vm, "$vm"_s))
    , m_dollarVMPrivateName(Identifier::fromUid(vm, &static_cast<SymbolImpl&>(Symbols::dollarVMPrivateName)))
    , m_polyProtoPrivateName(Identifier::fromUid(vm, &static_cast<SymbolImpl&>(Symbols::polyProtoPrivateName)))
{
    JSC_FOREACH_BUILTIN_FUNCTION_NAME(INITIALIZE_PUBLIC_TO_PRIVATE_ENTRY)
    JSC_COMMON_PRIVATE_IDENTIFIERS_EACH_PROPERTY_NAME(INITIALIZE_PUBLIC_TO_PRIVATE_ENTRY)
    JSC_COMMON_PRIVATE_IDENTIFIERS_EACH_WELL_KNOWN_SYMBOL(INITIALIZE_WELL_KNOWN_SYMBOL_PUBLIC_TO_PRIVATE_ENTRY)
    m_privateNameSet.add(static_cast<SymbolImpl*>(m_dollarVMPrivateName.impl()));
}

#undef INITIALIZE_BUILTIN_NAMES_IN_JSC
#undef INITIALIZE_BUILTIN_SYMBOLS_IN_JSC
#undef INITIALIZE_PUBLIC_TO_PRIVATE_ENTRY
#undef INITIALIZE_WELL_KNOWN_SYMBOL_PUBLIC_TO_PRIVATE_ENTRY


using LCharBuffer = WTF::HashTranslatorCharBuffer<LChar>;
using UCharBuffer = WTF::HashTranslatorCharBuffer<UChar>;

template<typename CharacterType>
struct CharBufferSeacher {
    using Buffer = WTF::HashTranslatorCharBuffer<CharacterType>;
    static unsigned hash(const Buffer& buf)
    {
        return buf.hash;
    }

    static bool equal(const String& str, const Buffer& buf)
    {
        return WTF::equal(str.impl(), buf.characters);
    }
};

template<typename CharacterType>
static PrivateSymbolImpl* lookUpPrivateNameImpl(const BuiltinNames::PrivateNameSet& set, const WTF::HashTranslatorCharBuffer<CharacterType>& buffer)
{
    auto iterator = set.find<CharBufferSeacher<CharacterType>>(buffer);
    if (iterator == set.end())
        return nullptr;
    StringImpl* impl = iterator->impl();
    ASSERT(impl->isSymbol());
    SymbolImpl* symbol = static_cast<SymbolImpl*>(impl);
    ASSERT(symbol->isPrivate());
    return static_cast<PrivateSymbolImpl*>(symbol);
}

template<typename CharacterType>
static SymbolImpl* lookUpWellKnownSymbolImpl(const BuiltinNames::WellKnownSymbolMap& map, const WTF::HashTranslatorCharBuffer<CharacterType>& buffer)
{
    auto iterator = map.find<CharBufferSeacher<CharacterType>>(buffer);
    if (iterator == map.end())
        return nullptr;
    return iterator->value;
}

PrivateSymbolImpl* BuiltinNames::lookUpPrivateName(std::span<const LChar> characters) const
{
    LCharBuffer buffer { characters };
    return lookUpPrivateNameImpl(m_privateNameSet, buffer);
}

PrivateSymbolImpl* BuiltinNames::lookUpPrivateName(std::span<const UChar> characters) const
{
    UCharBuffer buffer { characters };
    return lookUpPrivateNameImpl(m_privateNameSet, buffer);
}

PrivateSymbolImpl* BuiltinNames::lookUpPrivateName(const String& string) const
{
    if (string.is8Bit()) {
        LCharBuffer buffer { string.span8(), string.hash() };
        return lookUpPrivateNameImpl(m_privateNameSet, buffer);
    }
    UCharBuffer buffer { string.span16(), string.hash() };
    return lookUpPrivateNameImpl(m_privateNameSet, buffer);
}

SymbolImpl* BuiltinNames::lookUpWellKnownSymbol(std::span<const LChar> characters) const
{
    LCharBuffer buffer { characters };
    return lookUpWellKnownSymbolImpl(m_wellKnownSymbolsMap, buffer);
}

SymbolImpl* BuiltinNames::lookUpWellKnownSymbol(std::span<const UChar> characters) const
{
    UCharBuffer buffer { characters };
    return lookUpWellKnownSymbolImpl(m_wellKnownSymbolsMap, buffer);
}

SymbolImpl* BuiltinNames::lookUpWellKnownSymbol(const String& string) const
{
    if (string.is8Bit()) {
        LCharBuffer buffer { string.span8(), string.hash() };
        return lookUpWellKnownSymbolImpl(m_wellKnownSymbolsMap, buffer);
    }
    UCharBuffer buffer { string.span16(), string.hash() };
    return lookUpWellKnownSymbolImpl(m_wellKnownSymbolsMap, buffer);
}

} // namespace JSC
