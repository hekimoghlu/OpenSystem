/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 7, 2022.
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

#include "CallFrame.h"
#include "Identifier.h"
#include "Symbol.h"

namespace JSC  {

inline Identifier::Identifier(VM& vm, AtomStringImpl* string)
    : m_string(string)
{
#ifndef NDEBUG
    checkCurrentAtomStringTable(vm);
    if (string)
        ASSERT_WITH_MESSAGE(!string->length() || string->isSymbol() || AtomStringImpl::isInAtomStringTable(string), "The atomic string comes from an other thread!");
#else
    UNUSED_PARAM(vm);
#endif
}

inline Identifier::Identifier(VM& vm, const AtomString& string)
    : m_string(string)
{
#ifndef NDEBUG
    checkCurrentAtomStringTable(vm);
    if (!string.isNull())
        ASSERT_WITH_MESSAGE(!string.length() || string.impl()->isSymbol() || AtomStringImpl::isInAtomStringTable(string.impl()), "The atomic string comes from an other thread!");
#else
    UNUSED_PARAM(vm);
#endif
}

inline Ref<AtomStringImpl> Identifier::add(VM& vm, StringImpl* r)
{
#ifndef NDEBUG
    checkCurrentAtomStringTable(vm);
#endif
    return *AtomStringImpl::addWithStringTableProvider(vm, r);
}

inline Identifier Identifier::fromUid(VM& vm, UniquedStringImpl* uid)
{
    if (!uid || !uid->isSymbol())
        return Identifier(vm, uid);
    return static_cast<SymbolImpl&>(*uid);
}

inline Identifier Identifier::fromUid(const PrivateName& name)
{
    return name.uid();
}

inline Identifier Identifier::fromUid(SymbolImpl& symbol)
{
    return symbol;
}

ALWAYS_INLINE Identifier Identifier::fromString(VM& vm, ASCIILiteral s)
{
    return Identifier(vm, s);
}

inline Identifier Identifier::fromString(VM& vm, std::span<const LChar> s)
{
    return Identifier(vm, s);
}

inline Identifier Identifier::fromString(VM& vm, std::span<const UChar> s)
{
    return Identifier(vm, s);
}

inline Identifier Identifier::fromString(VM& vm, const String& string)
{
    return Identifier(vm, string.impl());
}

inline Identifier Identifier::fromString(VM& vm, AtomStringImpl* atomStringImpl)
{
    return Identifier(vm, atomStringImpl);
}

inline Identifier Identifier::fromString(VM& vm, Ref<AtomStringImpl>&& atomStringImpl)
{
    return Identifier(vm, WTFMove(atomStringImpl));
}

inline Identifier Identifier::fromString(VM& vm, const AtomString& atomString)
{
    return Identifier(vm, atomString);
}

inline Identifier Identifier::fromString(VM& vm, SymbolImpl* symbolImpl)
{
    return Identifier(vm, symbolImpl);
}

inline Identifier Identifier::fromLatin1(VM& vm, const char* s)
{
    return Identifier(vm, AtomString::fromLatin1(s));
}

inline JSValue identifierToJSValue(VM& vm, const Identifier& identifier)
{
    if (identifier.isSymbol())
        return Symbol::create(vm, static_cast<SymbolImpl&>(*identifier.impl()));
    return jsString(vm, identifier.string());
}

inline JSValue identifierToSafePublicJSValue(VM& vm, const Identifier& identifier) 
{
    if (identifier.isSymbol() && !identifier.isPrivateName())
        return Symbol::create(vm, static_cast<SymbolImpl&>(*identifier.impl()));
    return jsString(vm, identifier.string());
}

} // namespace JSC
