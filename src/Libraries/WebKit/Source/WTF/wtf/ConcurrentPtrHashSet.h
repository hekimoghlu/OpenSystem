/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 4, 2024.
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

#include <wtf/Atomics.h>
#include <wtf/FastMalloc.h>
#include <wtf/HashFunctions.h>
#include <wtf/Lock.h>
#include <wtf/Noncopyable.h>
#include <wtf/Vector.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace WTF {

// This is a concurrent hash-based set for pointers. It's optimized for:
//
// - High rate of contains() calls.
// - High rate of add() calls that don't add anything new. add() calls that don't add anything (nop adds)
//   don't mutate the table at all.
// - Not too many threads. I doubt this scales beyond ~4. Though, it may actually scale better than that
//   if the rate of nop adds is absurdly high.
//
// If we wanted this to scale better, the main change we'd have to make is how this table determines when
// to resize. Right now it's a shared counter. We lock;xadd this counter. One easy way to make that
// scalable is to require each thread that works with the ConcurrentPtrHashSet to register itself first.
// Then each thread would have some data structure that has a counter. We could institute the policy that
// each thread simply increments its own load counter, in its own data structure. Then, if any search to
// resolve a collision does more than N iterations, we can compute a combined load by summing the load
// counters of all of the thread data structures.
//
// ConcurrentPtrHashSet's main user, the GC, sees a 98% nop add rate in Speedometer. That's why this
// focuses so much on cases where the table does not change.
class ConcurrentPtrHashSet final {
    WTF_MAKE_NONCOPYABLE(ConcurrentPtrHashSet);
    WTF_MAKE_FAST_ALLOCATED;

public:
    WTF_EXPORT_PRIVATE ConcurrentPtrHashSet();
    WTF_EXPORT_PRIVATE ~ConcurrentPtrHashSet();
    
    template<typename T>
    bool contains(T value) const
    {
        return containsImpl(cast(value));
    }
    
    template<typename T>
    bool add(T value)
    {
        return addImpl(cast(value));
    }
    
    size_t size() const
    {
        Table* table = m_table.loadRelaxed();
        if (table == &m_stubTable)
            return sizeSlow();
        return table->load.loadRelaxed();
    }
    
    // Only call when you know that no other thread can call add(). This frees up memory without changing
    // the contents of the table.
    WTF_EXPORT_PRIVATE void deleteOldTables();
    
    // Only call when you know that no other thread can call add(). This frees up all memory except for
    // the smallest possible hashtable.
    WTF_EXPORT_PRIVATE void clear();
    
private:
    struct Table {
        WTF_MAKE_STRUCT_FAST_ALLOCATED;
        
        static std::unique_ptr<Table> create(unsigned size);
        void initializeStub();

        unsigned maxLoad() const { return size / 2; }

        // This can be any value >= 1 because the stub's size is 0, ensuring that
        // m_stubTable is always seen as "full". We choose 10 for no reason other
        // than it gives some warm fuzzies since it is greater than 1.
        static constexpr unsigned stubDefaultLoadValue = 10;

        unsigned size; // This is immutable.
        unsigned mask; // This is immutable.
        Atomic<unsigned> load;
        Atomic<void*> array[1];
    };
    
    static unsigned hash(const void* ptr)
    {
        return PtrHash<const void*>::hash(ptr);
    }
    
    void initialize();
    
    template<typename T>
    static void* cast(T value)
    {
        static_assert(sizeof(T) <= sizeof(void*), "type too big");
        union {
            void* ptr;
            T value;
        } u;
        u.ptr = nullptr;
        u.value = value;
        return u.ptr;
    }
    
    bool containsImpl(void* ptr) const
    {
        Table* table = m_table.loadRelaxed();
        if (table == &m_stubTable)
            return containsImplSlow(ptr);

        unsigned mask = table->mask;
        unsigned startIndex = hash(ptr) & mask;
        unsigned index = startIndex;
        for (;;) {
            void* entry = table->array[index].loadRelaxed();
            if (!entry)
                return false;
            if (entry == ptr)
                return true;
            index = (index + 1) & mask;
            RELEASE_ASSERT(index != startIndex);
        }
    }
    
    // Returns true if a new entry was added.
    bool addImpl(void* ptr)
    {
        Table* table = m_table.loadRelaxed();
        unsigned mask = table->mask;
        unsigned startIndex = hash(ptr) & mask;
        unsigned index = startIndex;
        for (;;) {
            void* entry = table->array[index].loadRelaxed();
            if (!entry)
                return addSlow(table, mask, startIndex, index, ptr);
            if (entry == ptr)
                return false;
            index = (index + 1) & mask;
            RELEASE_ASSERT(index != startIndex);
        }
    }
    
    WTF_EXPORT_PRIVATE bool addSlow(Table* table, unsigned mask, unsigned startIndex, unsigned index, void* ptr);
    WTF_EXPORT_PRIVATE bool containsImplSlow(void* ptr) const;
    WTF_EXPORT_PRIVATE size_t sizeSlow() const;

    void resizeIfNecessary();
    bool resizeAndAdd(void* ptr);

    Vector<std::unique_ptr<Table>, 4> m_allTables;
    Atomic<Table*> m_table; // This is never null.
    Table m_stubTable;
    mutable Lock m_lock; // We just use this to control resize races.
};

} // namespace WTF

using WTF::ConcurrentPtrHashSet;

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
