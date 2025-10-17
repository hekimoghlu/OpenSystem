/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 10, 2023.
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

#include "JSCJSValue.h"
#include <wtf/text/SymbolImpl.h>

namespace JSC {

class CodeBlock;
class Identifier;
class JSCell;

class CacheableIdentifier {
public:
    CacheableIdentifier() = default;

    static inline CacheableIdentifier createFromCell(JSCell* identifier);
    template <typename CodeBlockType>
    static inline CacheableIdentifier createFromIdentifierOwnedByCodeBlock(CodeBlockType*, const Identifier&);
    template <typename CodeBlockType>
    static inline CacheableIdentifier createFromIdentifierOwnedByCodeBlock(CodeBlockType*, UniquedStringImpl*);
    static inline CacheableIdentifier createFromImmortalIdentifier(UniquedStringImpl*);
    static inline CacheableIdentifier createFromSharedStub(UniquedStringImpl*);
    static constexpr CacheableIdentifier createFromRawBits(uintptr_t rawBits) { return CacheableIdentifier(rawBits); }

    CacheableIdentifier(const CacheableIdentifier&) = default;
    CacheableIdentifier(CacheableIdentifier&&) = default;

    CacheableIdentifier(std::nullptr_t)
        : m_bits(0)
    { }

    bool isUid() const { return m_bits & s_uidTag; }
    bool isCell() const { return !isUid(); }
    inline bool isSymbolCell() const;
    inline bool isStringCell() const;
    inline void ensureIsCell(VM&);

    bool isSymbol() const { return m_bits && uid()->isSymbol(); }
    bool isPrivateName() const { return isSymbol() && static_cast<SymbolImpl&>(*uid()).isPrivate(); }

    inline JSCell* cell() const;
    UniquedStringImpl* uid() const;

    explicit operator bool() const { return m_bits; }

    unsigned hash() const { return uid()->symbolAwareHash(); }

    CacheableIdentifier& operator=(const CacheableIdentifier&) = default;
    CacheableIdentifier& operator=(CacheableIdentifier&&) = default;

    bool operator==(const CacheableIdentifier&) const;
    bool operator==(const Identifier&) const;

    static inline bool isCacheableIdentifierCell(JSCell*);
    static inline bool isCacheableIdentifierCell(JSValue);

    uintptr_t rawBits() const { return m_bits; }

    template<typename Visitor> inline void visitAggregate(Visitor&) const;

    JS_EXPORT_PRIVATE void dump(PrintStream&) const;

private:
    explicit inline CacheableIdentifier(UniquedStringImpl*);
    explicit inline CacheableIdentifier(JSCell* identifier);
    explicit constexpr CacheableIdentifier(uintptr_t rawBits)
        : m_bits(rawBits)
    { }

    inline void setCellBits(JSCell*);
    inline void setUidBits(UniquedStringImpl*);

    // CacheableIdentifier can either hold a cell pointer or a uid. To discern which
    // it is holding, we tag the low bit if we have a uid. We choose to tag the uid
    // instead of the cell because this keeps the bits of the cell pointer form
    // unpolluted, and therefore, it can be scanned by our conservative GC to keep the
    // cell alive when the CacheableIdentifier is on the stack.
    static constexpr uintptr_t s_uidTag = 1;
    uintptr_t m_bits { 0 };
};

} // namespace JSC
