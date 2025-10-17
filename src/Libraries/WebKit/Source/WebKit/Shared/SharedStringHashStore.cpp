/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 16, 2022.
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
#include "SharedStringHashStore.h"

#include <algorithm>
#include <wtf/PageBlock.h>
#include <wtf/StdLibExtras.h>

namespace WebKit {

using namespace WebCore;

const unsigned sharedStringHashTableMaxLoad = 2;

static unsigned nextPowerOf2(unsigned v)
{
    // Taken from http://www.cs.utk.edu/~vose/c-stuff/bithacks.html
    // Devised by Sean Anderson, September 14, 2001

    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;

    return v;
}

static unsigned tableLengthForKeyCount(unsigned keyCount)
{
    // We want the table to be at least half empty.
    unsigned tableLength = nextPowerOf2(keyCount * sharedStringHashTableMaxLoad);

    // Ensure that the table length is at least the size of a page.
    size_t minimumTableLength = pageSize() / sizeof(SharedStringHash);
    if (tableLength < minimumTableLength)
        return minimumTableLength;

    return tableLength;
}

SharedStringHashStore::SharedStringHashStore(Client& client)
    : m_client(client)
    , m_pendingOperationsTimer(RunLoop::main(), this, &SharedStringHashStore::processPendingOperations)
{
}

std::optional<SharedMemory::Handle> SharedStringHashStore::createSharedMemoryHandle()
{
    return m_table.sharedMemory()->createHandle(SharedMemory::Protection::ReadOnly);
}

void SharedStringHashStore::scheduleAddition(SharedStringHash sharedStringHash)
{
    m_pendingOperations.append({ Operation::Add, sharedStringHash });

    if (!m_pendingOperationsTimer.isActive())
        m_pendingOperationsTimer.startOneShot(0_s);
}

void SharedStringHashStore::scheduleRemoval(WebCore::SharedStringHash sharedStringHash)
{
    m_pendingOperations.append({ Operation::Remove, sharedStringHash });

    if (!m_pendingOperationsTimer.isActive())
        m_pendingOperationsTimer.startOneShot(0_s);
}

bool SharedStringHashStore::contains(WebCore::SharedStringHash sharedStringHash)
{
    flushPendingChanges();
    return m_table.contains(sharedStringHash);
}

void SharedStringHashStore::clear()
{
    m_pendingOperationsTimer.stop();
    m_pendingOperations.clear();
    m_keyCount = 0;
    m_tableLength = 0;
    m_table.clear();
}

void SharedStringHashStore::flushPendingChanges()
{
    if (!m_pendingOperationsTimer.isActive())
        return;

    m_pendingOperationsTimer.stop();
    processPendingOperations();
}

void SharedStringHashStore::resizeTable(unsigned newTableLength)
{
    auto newTableMemory = SharedMemory::allocate((Checked<unsigned>(newTableLength) * sizeof(SharedStringHash)).value());
    if (!newTableMemory) {
        LOG_ERROR("Could not allocate shared memory for SharedStringHash table");
        return;
    }

    zeroSpan(newTableMemory->mutableSpan());

    RefPtr<SharedMemory> currentTableMemory = m_table.sharedMemory();
    unsigned currentTableLength = m_tableLength;

    m_table.setSharedMemory(newTableMemory.releaseNonNull());
    m_tableLength = newTableLength;

    if (currentTableMemory) {
        RELEASE_ASSERT(currentTableMemory->size() == (Checked<unsigned>(currentTableLength) * sizeof(SharedStringHash)).value());

        // Go through the current hash table and re-add all entries to the new hash table.
        auto currentSharedStringHashes = spanReinterpretCast<const SharedStringHash>(currentTableMemory->span());
        for (auto& sharedStringHash : currentSharedStringHashes) {
            if (!sharedStringHash)
                continue;

            bool didAddSharedStringHash = m_table.add(sharedStringHash);

            // It should always be possible to add the SharedStringHash to a new table.
            ASSERT_UNUSED(didAddSharedStringHash, didAddSharedStringHash);
        }
    }

    for (auto& operation : m_pendingOperations) {
        switch (operation.type) {
        case Operation::Add:
            if (m_table.add(operation.sharedStringHash))
                ++m_keyCount;
            break;
        case Operation::Remove:
            if (m_table.remove(operation.sharedStringHash))
                --m_keyCount;
            break;
        }
    }
    m_pendingOperations.clear();

    m_client.didInvalidateSharedMemory();
}

void SharedStringHashStore::processPendingOperations()
{
    unsigned currentTableLength = m_tableLength;
    unsigned approximateNewHashCount = std::count_if(m_pendingOperations.begin(), m_pendingOperations.end(), [](auto& operation) {
        return operation.type == Operation::Add;
    });
    // FIXME: The table can currently only grow. We should probably support shrinking it to save memory.
    unsigned newTableLength = tableLengthForKeyCount(m_keyCount + approximateNewHashCount);

    newTableLength = std::max(currentTableLength, newTableLength);

    if (currentTableLength != newTableLength) {
        resizeTable(newTableLength);
        return;
    }

    Vector<SharedStringHash> addedSharedStringHashes;
    Vector<SharedStringHash> removedSharedStringHashes;
    addedSharedStringHashes.reserveInitialCapacity(approximateNewHashCount);
    removedSharedStringHashes.reserveInitialCapacity(m_pendingOperations.size() - approximateNewHashCount);
    for (auto& operation : m_pendingOperations) {
        switch (operation.type) {
        case Operation::Add:
            if (m_table.add(operation.sharedStringHash)) {
                addedSharedStringHashes.append(operation.sharedStringHash);
                ++m_keyCount;
            }
            break;
        case Operation::Remove:
            if (m_table.remove(operation.sharedStringHash)) {
                removedSharedStringHashes.append(operation.sharedStringHash);
                --m_keyCount;
            }
            break;
        }
    }

    m_pendingOperations.clear();

    if (!addedSharedStringHashes.isEmpty() || !removedSharedStringHashes.isEmpty())
        m_client.didUpdateSharedStringHashes(addedSharedStringHashes, removedSharedStringHashes);
}

} // namespace WebKit
