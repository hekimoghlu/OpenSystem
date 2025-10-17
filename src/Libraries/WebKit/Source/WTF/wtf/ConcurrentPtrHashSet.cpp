/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 30, 2024.
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
#include <wtf/ConcurrentPtrHashSet.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace WTF {

ConcurrentPtrHashSet::ConcurrentPtrHashSet()
{
    initialize();
}

ConcurrentPtrHashSet::~ConcurrentPtrHashSet() = default;

void ConcurrentPtrHashSet::deleteOldTables()
{
    // This is just in case. It does not make it OK for other threads to call add(). But it might prevent
    // some bad crashes if we did make that mistake.
    Locker locker { m_lock };

    ASSERT(m_table.loadRelaxed() != &m_stubTable);
    m_allTables.removeAllMatching(
        [&] (std::unique_ptr<Table>& table) -> bool {
            return table.get() != m_table.loadRelaxed();
        });
}

void ConcurrentPtrHashSet::clear()
{
    // This is just in case. It does not make it OK for other threads to call add(). But it might prevent
    // some bad crashes if we did make that mistake.
    Locker locker { m_lock };
    
    m_allTables.clear();
    initialize();
}

void ConcurrentPtrHashSet::initialize()
{
    constexpr unsigned initialSize = 32;
    std::unique_ptr<Table> table = Table::create(initialSize);
    m_table.storeRelaxed(table.get());
    m_allTables.append(WTFMove(table));
    m_stubTable.initializeStub();
}

bool ConcurrentPtrHashSet::addSlow(Table* table, unsigned mask, unsigned startIndex, unsigned index, void* ptr)
{
    if (table->load.exchangeAdd(1) >= table->maxLoad())
        return resizeAndAdd(ptr);
    
    for (;;) {
        void* oldEntry = table->array[index].compareExchangeStrong(nullptr, ptr);
        if (!oldEntry) {
            if (m_table.load() != table) {
                // We added an entry to an old table! We need to reexecute the add on the new table.
                return add(ptr);
            }
            return true;
        }
        if (oldEntry == ptr)
            return false;
        index = (index + 1) & mask;
        RELEASE_ASSERT(index != startIndex);
    }
}

bool ConcurrentPtrHashSet::containsImplSlow(void* ptr) const
{
    Locker locker { m_lock };
    ASSERT(m_table.loadRelaxed() != &m_stubTable);
    return containsImpl(ptr);
}

size_t ConcurrentPtrHashSet::sizeSlow() const
{
    Locker locker { m_lock };
    ASSERT(m_table.loadRelaxed() != &m_stubTable);
    return size();
}

void ConcurrentPtrHashSet::resizeIfNecessary()
{
    Locker locker { m_lock };
    Table* table = m_table.loadRelaxed();
    ASSERT(table != &m_stubTable);
    if (table->load.loadRelaxed() < table->maxLoad())
        return;

    // Stubbing out m_table with m_stubTable here is necessary to ensure that
    // we don't miss copying any entries that may be concurrently be added.
    //
    // If addSlow() completes before this stubbing, the new entry is guaranteed
    // to be copied below.
    //
    // If addSlow() completes after this stubbing, addSlow()  will check m_table
    // before it finishes, and detect that its newly added entry may not have
    // made it in. As a result, it will try to re-add the entry, and end up
    // blocking on resizeIfNecessary() until the resizing is donw. This is
    // because m_stubTable will tell addSlow() think that the table is out of
    // space and it needs to resize. NOTE: m_stubTable always says it is out of
    // space.
    m_table.store(&m_stubTable);

    std::unique_ptr<Table> newTable = Table::create(table->size * 2);
    unsigned mask = newTable->mask;
    unsigned load = 0;
    for (unsigned i = 0; i < table->size; ++i) {
        void* ptr = table->array[i].loadRelaxed();
        if (!ptr)
            continue;
        
        unsigned startIndex = hash(ptr) & mask;
        unsigned index = startIndex;
        for (;;) {
            Atomic<void*>& entryRef = newTable->array[index];
            void* entry = entryRef.loadRelaxed();
            if (!entry) {
                entryRef.storeRelaxed(ptr);
                break;
            }
            RELEASE_ASSERT(entry != ptr);
            index = (index + 1) & mask;
            RELEASE_ASSERT(index != startIndex);
        }
        
        load++;
    }
    
    newTable->load.storeRelaxed(load);

    m_table.store(newTable.get());

    // addSlow() will always start by exchangeAdd'ing 1 to the current m_table's
    // load value before checking if it exceeds its max allowed load. For the
    // real m_table, this is not an issue because at most, it will accummulate
    // up to N extra adds above max load, where N is the number of threads
    // concurrrently adding entries.
    //
    // However, m_table may be replaced with m_stubTable for each resize
    // operation. As a result, the cummulative error on its load value
    // may far exceed N (as specified above). To fix this, we always reset it
    // here to prevent an overflow. Note: a load of stubDefaultLoadValue means
    // that m_stubTable is full since its size is 0.
    //
    // In practice, this won't matter because we most likely won't do so many
    // resize operations such that this will get to the point of overflowing.
    // However, since resizing is not in the fast path, let's just be pedantic
    // and reset it for correctness.
    m_stubTable.load.store(Table::stubDefaultLoadValue);

    m_allTables.append(WTFMove(newTable));
}

bool ConcurrentPtrHashSet::resizeAndAdd(void* ptr)
{
    resizeIfNecessary();
    return add(ptr);
}

std::unique_ptr<ConcurrentPtrHashSet::Table> ConcurrentPtrHashSet::Table::create(unsigned size)
{
    std::unique_ptr<ConcurrentPtrHashSet::Table> result(new (fastMalloc(OBJECT_OFFSETOF(Table, array) + sizeof(Atomic<void*>) * size)) Table());
    result->size = size;
    result->mask = size - 1;
    result->load.storeRelaxed(0);
    for (unsigned i = 0; i < size; ++i)
        result->array[i].storeRelaxed(nullptr);
    return result;
}

void ConcurrentPtrHashSet::Table::initializeStub()
{
    // The stub table is set up to look like it is already filled up. This is
    // so that it can be used during resizing to force all attempts to add to
    // be routed to resizeAndAdd() where it will block until the resizing is
    // done.
    size = 0;
    mask = 0;
    load.storeRelaxed(stubDefaultLoadValue);
    array[0].storeRelaxed(nullptr);
}

} // namespace WTF

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
