/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 14, 2024.
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

#include "CommonIdentifiers.h"
#include "Identifier.h"
#include "MathCommon.h"
#include <array>
#include <type_traits>
#include <wtf/SegmentedVector.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

    class ParserArenaDeletable;

    DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(IdentifierArena);
    class IdentifierArena {
        WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(IdentifierArena);
    public:
        IdentifierArena()
        {
            clear();
        }

        template <typename T>
        ALWAYS_INLINE const Identifier& makeIdentifier(VM&, std::span<const T> characters);
        ALWAYS_INLINE const Identifier& makeEmptyIdentifier(VM&);
        ALWAYS_INLINE const Identifier& makeIdentifierLCharFromUChar(VM&, std::span<const UChar> characters);
        ALWAYS_INLINE const Identifier& makeIdentifier(VM&, SymbolImpl*);

        const Identifier* makeBigIntDecimalIdentifier(VM&, const Identifier&, uint8_t radix);
        const Identifier& makeNumericIdentifier(VM&, double number);
        const Identifier& makePrivateIdentifier(VM&, ASCIILiteral, unsigned);

    public:
        static const int MaximumCachableCharacter = 128;
        typedef SegmentedVector<Identifier, 64> IdentifierVector;
        void clear()
        {
            m_identifiers.clear();
            for (int i = 0; i < MaximumCachableCharacter; i++)
                m_shortIdentifiers[i] = nullptr;
            for (int i = 0; i < MaximumCachableCharacter; i++)
                m_recentIdentifiers[i] = nullptr;
        }

    private:
        IdentifierVector m_identifiers;
        std::array<Identifier*, MaximumCachableCharacter> m_shortIdentifiers;
        std::array<Identifier*, MaximumCachableCharacter> m_recentIdentifiers;
    };

    template <typename T>
    ALWAYS_INLINE const Identifier& IdentifierArena::makeIdentifier(VM& vm, std::span<const T> characters)
    {
        if (characters.empty())
            return vm.propertyNames->emptyIdentifier;
        if (characters.front() >= MaximumCachableCharacter) {
            m_identifiers.append(Identifier::fromString(vm, characters));
            return m_identifiers.last();
        }
        if (characters.size() == 1) {
            if (Identifier* ident = m_shortIdentifiers[characters.front()])
                return *ident;
            m_identifiers.append(Identifier::fromString(vm, characters));
            m_shortIdentifiers[characters.front()] = &m_identifiers.last();
            return m_identifiers.last();
        }
        Identifier* ident = m_recentIdentifiers[characters.front()];
        if (ident && Identifier::equal(ident->impl(), characters))
            return *ident;
        m_identifiers.append(Identifier::fromString(vm, characters));
        m_recentIdentifiers[characters.front()] = &m_identifiers.last();
        return m_identifiers.last();
    }

    ALWAYS_INLINE const Identifier& IdentifierArena::makeIdentifier(VM&, SymbolImpl* symbol)
    {
        ASSERT(symbol);
        m_identifiers.append(Identifier::fromUid(*symbol));
        return m_identifiers.last();
    }

    ALWAYS_INLINE const Identifier& IdentifierArena::makeEmptyIdentifier(VM& vm)
    {
        return vm.propertyNames->emptyIdentifier;
    }

    ALWAYS_INLINE const Identifier& IdentifierArena::makeIdentifierLCharFromUChar(VM& vm, std::span<const UChar> characters)
    {
        if (characters.empty())
            return vm.propertyNames->emptyIdentifier;
        if (characters.front() >= MaximumCachableCharacter) {
            m_identifiers.append(Identifier::createLCharFromUChar(vm, characters));
            return m_identifiers.last();
        }
        if (characters.size() == 1) {
            if (Identifier* ident = m_shortIdentifiers[characters.front()])
                return *ident;
            m_identifiers.append(Identifier::fromString(vm, characters));
            m_shortIdentifiers[characters.front()] = &m_identifiers.last();
            return m_identifiers.last();
        }
        Identifier* ident = m_recentIdentifiers[characters.front()];
        if (ident && Identifier::equal(ident->impl(), characters))
            return *ident;
        m_identifiers.append(Identifier::createLCharFromUChar(vm, characters));
        m_recentIdentifiers[characters.front()] = &m_identifiers.last();
        return m_identifiers.last();
    }
    
    inline const Identifier& IdentifierArena::makeNumericIdentifier(VM& vm, double number)
    {
        Identifier token;
        // This is possible that number can be -0, but it is OK since ToString(-0) is "0".
        if (canBeInt32(number))
            token = Identifier::from(vm, static_cast<int32_t>(number));
        else
            token = Identifier::from(vm, number);
        m_identifiers.append(WTFMove(token));
        return m_identifiers.last();
    }

    DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(ParserArena);

    class ParserArena {
        WTF_MAKE_NONCOPYABLE(ParserArena);
    public:
        ParserArena();
        ~ParserArena();

        void swap(ParserArena& otherArena)
        {
            std::swap(m_freeableMemory, otherArena.m_freeableMemory);
            std::swap(m_freeablePoolEnd, otherArena.m_freeablePoolEnd);
            m_identifierArena.swap(otherArena.m_identifierArena);
            m_freeablePools.swap(otherArena.m_freeablePools);
            m_deletableObjects.swap(otherArena.m_deletableObjects);
        }

        void* allocateFreeable(size_t size)
        {
            ASSERT(size);
            ASSERT(size <= freeablePoolSize);
            size_t alignedSize = alignSize(size);
            ASSERT(alignedSize <= freeablePoolSize);
            if (UNLIKELY(static_cast<size_t>(m_freeablePoolEnd - m_freeableMemory) < alignedSize))
                allocateFreeablePool();
            void* block = m_freeableMemory;
            m_freeableMemory += alignedSize;
            return block;
        }

        template<typename T, typename = std::enable_if_t<std::is_base_of<ParserArenaDeletable, T>::value>>
        void* allocateDeletable(size_t size)
        {
            // T may extend ParserArenaDeletable via multiple inheritance, but not as T's first
            // base class. m_deletableObjects is expecting pointers to objects of the shape of
            // ParserArenaDeletable. We ensure this by allocating T, and casting it to
            // ParserArenaDeletable to get the correct pointer to append to m_deletableObjects.
            T* instance = static_cast<T*>(allocateFreeable(size));
            ParserArenaDeletable* deletable = static_cast<ParserArenaDeletable*>(instance);
            m_deletableObjects.append(deletable);
            return instance;
        }

        IdentifierArena& identifierArena()
        {
            if (UNLIKELY (!m_identifierArena))
                m_identifierArena = makeUnique<IdentifierArena>();
            return *m_identifierArena;
        }

    private:
        static const size_t freeablePoolSize = 8000;

        static size_t alignSize(size_t size)
        {
            return (size + sizeof(WTF::AllocAlignmentInteger) - 1) & ~(sizeof(WTF::AllocAlignmentInteger) - 1);
        }

        void* freeablePool();
        void allocateFreeablePool();
        void deallocateObjects();

        char* m_freeableMemory;
        char* m_freeablePoolEnd;

        std::unique_ptr<IdentifierArena> m_identifierArena;
        Vector<void*> m_freeablePools;
        Vector<ParserArenaDeletable*> m_deletableObjects;
    };

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
