/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 23, 2024.
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

#include "CacheableIdentifier.h"

#include "Identifier.h"
#include "JSCJSValueInlines.h"
#include "JSCell.h"
#include "VM.h"
#include <wtf/text/UniquedStringImpl.h>

namespace JSC {

template <typename CodeBlockType>
inline CacheableIdentifier CacheableIdentifier::createFromIdentifierOwnedByCodeBlock(CodeBlockType* codeBlock, const Identifier& i)
{
    return createFromIdentifierOwnedByCodeBlock(codeBlock, i.impl());
}

template <typename CodeBlockType>
inline CacheableIdentifier CacheableIdentifier::createFromIdentifierOwnedByCodeBlock(CodeBlockType* codeBlock, UniquedStringImpl* uid)
{
    ASSERT_UNUSED(codeBlock, codeBlock->hasIdentifier(uid));
    return CacheableIdentifier(uid);
}

inline CacheableIdentifier CacheableIdentifier::createFromImmortalIdentifier(UniquedStringImpl* uid)
{
    return CacheableIdentifier(uid);
}

inline CacheableIdentifier CacheableIdentifier::createFromSharedStub(UniquedStringImpl* uid)
{
    return CacheableIdentifier(uid);
}

inline CacheableIdentifier CacheableIdentifier::createFromCell(JSCell* i)
{
    return CacheableIdentifier(i);
}

inline CacheableIdentifier::CacheableIdentifier(UniquedStringImpl* uid)
{
    setUidBits(uid);
}

inline CacheableIdentifier::CacheableIdentifier(JSCell* identifier)
{
    ASSERT(isCacheableIdentifierCell(identifier));
    setCellBits(identifier);
}

inline JSCell* CacheableIdentifier::cell() const
{
    ASSERT(isCell());
    return std::bit_cast<JSCell*>(m_bits);
}

inline UniquedStringImpl* CacheableIdentifier::uid() const
{
    if (!m_bits)
        return nullptr;
    if (isUid())
        return std::bit_cast<UniquedStringImpl*>(m_bits & ~s_uidTag);
    if (isSymbolCell())
        return &jsCast<Symbol*>(cell())->uid();
    ASSERT(isStringCell());
    JSString* string = jsCast<JSString*>(cell());
    return std::bit_cast<UniquedStringImpl*>(string->getValueImpl());
}

inline bool CacheableIdentifier::isCacheableIdentifierCell(JSCell* cell)
{
    if (cell->isSymbol())
        return true;
    if (!cell->isString())
        return false;
    JSString* string = jsCast<JSString*>(cell);
    if (const StringImpl* impl = string->tryGetValueImpl())
        return impl->isAtom();
    return false;
}

inline bool CacheableIdentifier::isCacheableIdentifierCell(JSValue value)
{
    if (!value.isCell())
        return false;
    return isCacheableIdentifierCell(value.asCell());
}

inline bool CacheableIdentifier::isSymbolCell() const
{
    return isCell() && cell()->isSymbol();
}

inline bool CacheableIdentifier::isStringCell() const
{
    return isCell() && cell()->isString();
}

inline void CacheableIdentifier::ensureIsCell(VM& vm)
{
    if (!isCell()) {
        if (uid()->isSymbol())
            setCellBits(Symbol::create(vm, static_cast<SymbolImpl&>(*uid())));
        else
            setCellBits(jsString(vm, String(static_cast<AtomStringImpl*>(uid()))));
    }
    ASSERT(isCell());
}

inline void CacheableIdentifier::setCellBits(JSCell* cell)
{
    RELEASE_ASSERT(isCacheableIdentifierCell(cell));
    m_bits = std::bit_cast<uintptr_t>(cell);
}

inline void CacheableIdentifier::setUidBits(UniquedStringImpl* uid)
{
    m_bits = std::bit_cast<uintptr_t>(uid) | s_uidTag;
}

template<typename Visitor>
inline void CacheableIdentifier::visitAggregate(Visitor& visitor) const
{
    if (m_bits && isCell())
        visitor.appendUnbarriered(cell());
}

inline bool CacheableIdentifier::operator==(const CacheableIdentifier& other) const
{
    return uid() == other.uid();
}

inline bool CacheableIdentifier::operator==(const Identifier& other) const
{
    return uid() == other.impl();
}

} // namespace JSC
