/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 8, 2023.
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

#include "HashMapHelper.h"
#include "WeakMapImpl.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

ALWAYS_INLINE uint32_t jsWeakMapHash(JSCell* key)
{
    return wangsInt64Hash(JSValue::encode(key));
}

ALWAYS_INLINE uint32_t nextCapacityAfterBatchRemoval(uint32_t capacity, uint32_t keyCount)
{
    while (shouldShrink(capacity, keyCount))
        capacity = nextCapacity(capacity, keyCount);
    return capacity;
}

static ALWAYS_INLINE bool canBeHeldWeakly(JSValue value)
{
    // https://tc39.es/proposal-symbols-as-weakmap-keys/#sec-canbeheldweakly-abstract-operation
    if (value.isObject())
        return true;
    if (!value.isSymbol())
        return false;
    return !asSymbol(value)->uid().isRegistered();
}

template <typename WeakMapBucket>
ALWAYS_INLINE void WeakMapImpl<WeakMapBucket>::add(VM& vm, JSCell* key, JSValue value)
{
    DisallowGC disallowGC;
    add(vm, key, value, jsWeakMapHash(key));
}

template <typename WeakMapBucket>
ALWAYS_INLINE void WeakMapImpl<WeakMapBucket>::add(VM& vm, JSCell* key, JSValue value, uint32_t hash)
{
    DisallowGC disallowGC;
    ASSERT_WITH_MESSAGE(jsWeakMapHash(key) == hash, "We expect hash value is what we expect.");

    addInternal(vm, key, value, hash);
    if (shouldRehashAfterAdd())
        rehash();
}

template <typename WeakMapBucket>
ALWAYS_INLINE void WeakMapImpl<WeakMapBucket>::addBucket(VM& vm, JSCell* key, JSValue value, uint32_t hash, size_t index)
{
    UNUSED_PARAM(hash);
    ASSERT(jsWeakMapHash(key) == hash);
    ASSERT(!findBucket(key, hash));

    WeakMapBucket* newEntry = buffer() + index;
    ASSERT(newEntry);
    ASSERT(newEntry->isEmpty());

    newEntry->setKey(vm, this, key);
    newEntry->setValue(vm, this, value);
    ++m_keyCount;

    if (shouldRehashAfterAdd())
        rehash();
}

// Note that this function can be executed in parallel as long as the mutator stops.
template<typename WeakMapBucket>
void WeakMapImpl<WeakMapBucket>::finalizeUnconditionally(VM& vm, CollectionScope)
{
    auto* buffer = this->buffer();
    for (uint32_t index = 0; index < m_capacity; ++index) {
        auto* bucket = buffer + index;
        if (bucket->isEmpty() || bucket->isDeleted())
            continue;

        if (vm.heap.isMarked(bucket->key()))
            continue;

        bucket->makeDeleted();
        ++m_deleteCount;
        RELEASE_ASSERT(m_keyCount > 0);
        --m_keyCount;
    }

    if (shouldShrink())
        rehash(RehashMode::RemoveBatching);
}

template<typename WeakMapBucket>
void WeakMapImpl<WeakMapBucket>::rehash(RehashMode mode)
{
    // Since shrinking is done just after GC runs (finalizeUnconditionally), WeakMapImpl::rehash()
    // function must not touch any GC related features. This is why we do not allocate WeakMapBuffer
    // in auxiliary buffer.

    uint32_t oldCapacity = m_capacity;
    MallocPtr<WeakMapBufferType> oldBuffer = WTFMove(m_buffer);

    uint32_t capacity = m_capacity;
    if (mode == RehashMode::RemoveBatching) {
        ASSERT(shouldShrink());
        capacity = nextCapacityAfterBatchRemoval(capacity, m_keyCount);
    } else
        capacity = nextCapacity(capacity, m_keyCount);
    makeAndSetNewBuffer(capacity);

    auto* buffer = this->buffer();
    const uint32_t mask = m_capacity - 1;
    for (uint32_t oldIndex = 0; oldIndex < oldCapacity; ++oldIndex) {
        auto* entry = oldBuffer->buffer() + oldIndex;
        if (entry->isEmpty() || entry->isDeleted())
            continue;

        uint32_t index = jsWeakMapHash(entry->key()) & mask;
        WeakMapBucket* bucket = buffer + index;
        while (!bucket->isEmpty()) {
            index = (index + 1) & mask;
            bucket = buffer + index;
        }
        bucket->copyFrom(*entry);
    }

    m_deleteCount = 0;

    checkConsistency();
}

template<typename WeakMapBucket>
ALWAYS_INLINE uint32_t WeakMapImpl<WeakMapBucket>::shouldRehashAfterAdd() const
{
    return JSC::shouldRehash(m_capacity, m_keyCount, m_deleteCount);
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
