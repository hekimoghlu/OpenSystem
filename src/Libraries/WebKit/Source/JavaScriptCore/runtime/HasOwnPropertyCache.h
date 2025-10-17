/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 2, 2024.
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

#include "JSObject.h"
#include "PropertySlot.h"
#include "Structure.h"
#include <wtf/UniqueRef.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(HasOwnPropertyCache);

class alignas(8) HasOwnPropertyCache {
    static const uint32_t size = 2 * 1024;
    static_assert(hasOneBitSet(size), "size should be a power of two.");
public:
    static const uint32_t mask = size - 1;

    struct Entry {
        static constexpr ptrdiff_t offsetOfStructureID() { return OBJECT_OFFSETOF(Entry, structureID); }
        static constexpr ptrdiff_t offsetOfImpl() { return OBJECT_OFFSETOF(Entry, impl); }
        static constexpr ptrdiff_t offsetOfResult() { return OBJECT_OFFSETOF(Entry, result); }

        RefPtr<UniquedStringImpl> impl;
        StructureID structureID;
        bool result { false };
    };

    HasOwnPropertyCache() = delete;

    void operator delete(void* cache)
    {
        static_cast<HasOwnPropertyCache*>(cache)->clear();
        HasOwnPropertyCacheMalloc::free(cache);
    }

    static UniqueRef<HasOwnPropertyCache> create()
    {
        size_t allocationSize = sizeof(Entry) * size;
        HasOwnPropertyCache* result = static_cast<HasOwnPropertyCache*>(HasOwnPropertyCacheMalloc::malloc(allocationSize));
        result->clearBuffer();
        return UniqueRef { *result };
    }

    ALWAYS_INLINE static uint32_t hash(StructureID structureID, UniquedStringImpl* impl)
    {
        return std::bit_cast<uint32_t>(structureID) + impl->hash();
    }

    ALWAYS_INLINE std::optional<bool> get(Structure* structure, PropertyName propName)
    {
        UniquedStringImpl* impl = propName.uid();
        StructureID id = structure->id();
        uint32_t index = HasOwnPropertyCache::hash(id, impl) & mask;
        Entry& entry = std::bit_cast<Entry*>(this)[index];
        if (entry.structureID == id && entry.impl.get() == impl)
            return entry.result;
        return std::nullopt;
    }

    ALWAYS_INLINE void tryAdd(PropertySlot& slot, JSObject* object, PropertyName propName, bool result)
    {
        if (parseIndex(propName))
            return;

        if (!slot.isCacheable() && !slot.isUnset())
            return;

        if (object->type() == GlobalProxyType)
            return;

        Structure* structure = object->structure();
        if (!structure->typeInfo().prohibitsPropertyCaching()
            && structure->propertyAccessesAreCacheable()
            && (!slot.isUnset() || structure->propertyAccessesAreCacheableForAbsence())) {
            if (structure->isDictionary()) {
                // FIXME: We should be able to flatten a dictionary object again.
                // https://bugs.webkit.org/show_bug.cgi?id=163092
                return;
            }

            ASSERT(!result == slot.isUnset());

            UniquedStringImpl* impl = propName.uid();
            StructureID id = structure->id();
            uint32_t index = HasOwnPropertyCache::hash(id, impl) & mask;
            std::bit_cast<Entry*>(this)[index] = Entry { RefPtr<UniquedStringImpl>(impl), id, result };
        }
    }

    void clear()
    {
        Entry* buffer = std::bit_cast<Entry*>(this);
        for (uint32_t i = 0; i < size; ++i)
            buffer[i].Entry::~Entry();

        clearBuffer();
    }

private:
    void clearBuffer()
    {
        Entry* buffer = std::bit_cast<Entry*>(this);
        for (uint32_t i = 0; i < size; ++i)
            new (&buffer[i]) Entry();
    }
};

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
