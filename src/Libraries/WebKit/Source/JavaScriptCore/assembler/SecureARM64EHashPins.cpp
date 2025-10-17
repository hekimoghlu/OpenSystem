/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 11, 2023.
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
#include "SecureARM64EHashPins.h"

#include "ExecutableAllocator.h"
#include "FastJITPermissions.h"
#include "SecureARM64EHashPinsInlines.h"
#include <wtf/Condition.h>
#include <wtf/Lock.h>

namespace JSC {

#if CPU(ARM64E) && ENABLE(JIT)

struct WriteToJITRegionScope {
    ALWAYS_INLINE WriteToJITRegionScope()
    {
        threadSelfRestrict<MemoryRestriction::kRwxToRw>();
    }

    ALWAYS_INLINE ~WriteToJITRegionScope()
    {
        threadSelfRestrict<MemoryRestriction::kRwxToRx>();
    }
};

struct ValidateNonReentrancyScope {
    ALWAYS_INLINE ValidateNonReentrancyScope(SecureARM64EHashPins::Metadata* metadata)
        : m_metadata(metadata)
    {
        uint32_t validateNonReentrancy = m_metadata->assertNotReentrant.exchange(1, std::memory_order_seq_cst);
        RELEASE_ASSERT(!validateNonReentrancy);
    }

    ALWAYS_INLINE ~ValidateNonReentrancyScope()
    {
        uint32_t validateNonReentrancy = m_metadata->assertNotReentrant.exchange(0, std::memory_order_seq_cst);
        RELEASE_ASSERT(validateNonReentrancy == 1);
    }
    SecureARM64EHashPins::Metadata* m_metadata;
};

// This class is allocated once per process, so static lock is ok.
static Lock hashPinsLock;

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

ALWAYS_INLINE static ExecutableMemoryHandle::MemoryPtr allocateInExecutableMemory(size_t size)
{
    ExecutableMemoryHandle* handle = ExecutableAllocator::singleton().allocate(size, JITCompilationMustSucceed).leakRef();
    RELEASE_ASSERT(handle);
    void* memory = handle->start().untaggedPtr<char*>();
    RELEASE_ASSERT(isJITPC(memory) && isJITPC(std::bit_cast<char*>(memory) + size - 1));
    return handle->start();
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

ALWAYS_INLINE SecureARM64EHashPins::Page::Page()
{
    for (unsigned i = 0; i < numEntriesPerPage; ++i) {
        auto& entry = this->entry(i);
        entry.pin = 0;
        entry.key = 0;
    }
}

static ALWAYS_INLINE void initializePage(const WriteToJITRegionScope&, SecureARM64EHashPins::Page* page)
{
    new (page) SecureARM64EHashPins::Page;
}

#define VALIDATE_THIS_VALUE() RELEASE_ASSERT(this == &g_jscConfig.arm64eHashPins) 

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

ALWAYS_INLINE auto SecureARM64EHashPins::metadata() -> Metadata*
{
    VALIDATE_THIS_VALUE();
    return std::bit_cast<Metadata*>(m_memory) - 1;
}

void SecureARM64EHashPins::initializeAtStartup()
{
    if (!g_jscConfig.useFastJITPermissions)
        return;

    VALIDATE_THIS_VALUE();
    RELEASE_ASSERT(!m_memory);

    m_memory = allocateInExecutableMemory(sizeof(Metadata) + Page::allocationSize()).untaggedPtr<char*>() + sizeof(Metadata);

    {
        WriteToJITRegionScope writeScope;

        new (metadata()) Metadata; 
        initializePage(writeScope, firstPage());
    }
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

bool SecureARM64EHashPins::allocatePinForCurrentThreadImpl(const AbstractLocker&)
{
    VALIDATE_THIS_VALUE();

    size_t newEntryIndex;
    Page* newEntryPage;

    bool found = false;
    size_t baseIndex = 0;
    forEachPage([&] (Page& page) {
        size_t bit = page.findClearBit();
        if (bit < numEntriesPerPage) {
            found = true;
            newEntryIndex = baseIndex + bit;
            newEntryPage = &page;
            return IterationStatus::Done;
        }

        baseIndex += numEntriesPerPage;
        return IterationStatus::Continue;
    });

    if (!found)
        return false;

    auto findResult = findFirstEntry();
    Entry* preexistingEntry = findResult.entry;
    size_t preexistingEntryIndex = findResult.index;
    Page* preexistingEntryPage = findResult.page;

    {
        auto* metadata = this->metadata();

        WriteToJITRegionScope writeScope;
        ValidateNonReentrancyScope validateNonReentrancy(metadata);

        RELEASE_ASSERT(isJITPC(metadata));
        uint64_t nextPin = metadata->nextPin.exchangeAdd(1, std::memory_order_seq_cst);
        RELEASE_ASSERT(nextPin);

        newEntryPage->setIsInUse(newEntryIndex);

        if (preexistingEntry && preexistingEntryIndex < newEntryIndex) {
            RELEASE_ASSERT(preexistingEntryPage->isInUse(preexistingEntryIndex));

            newEntryPage->entry(newEntryIndex) = *preexistingEntry;
            newEntryIndex = preexistingEntryIndex;
            newEntryPage = preexistingEntryPage;
        }

        auto& entry = newEntryPage->entry(newEntryIndex);
        entry.pin = nextPin;
        entry.key = keyForCurrentThread();
    }

    return true;
}

void SecureARM64EHashPins::allocatePinForCurrentThread()
{
    if (!g_jscConfig.useFastJITPermissions)
        return;

    VALIDATE_THIS_VALUE();

    Locker locker { hashPinsLock };

    if (allocatePinForCurrentThreadImpl(locker))
        return;

    // Allocate a new page
    {
        Page* lastPage = firstPage();
        while (lastPage->next)
            lastPage = lastPage->next;
        RELEASE_ASSERT(!lastPage->next);

        auto newPage = allocateInExecutableMemory(Page::allocationSize());

        {
            WriteToJITRegionScope writeScope;
            ValidateNonReentrancyScope validateNonReentrancy(metadata());
            initializePage(writeScope, newPage.untaggedPtr<Page*>());
            // This can be read from concurrently. Make sure it's in a good state for that to happen.
            WTF::storeStoreFence();
            lastPage->next = newPage.untaggedPtr<Page*>();
        }
    }

    bool success = allocatePinForCurrentThreadImpl(locker);
    RELEASE_ASSERT(success);
}

void SecureARM64EHashPins::deallocatePinForCurrentThread()
{
    if (!g_jscConfig.useFastJITPermissions)
        return;

    VALIDATE_THIS_VALUE();

    Locker locker { hashPinsLock };

    auto clear = [] (Entry& entry) {
        entry.pin = 0;
        entry.key = 0;
    };

    Page* removalPage;
    size_t removalIndex;
    {
        auto findResult = findFirstEntry();
        Entry& entry = *findResult.entry;
        removalIndex = findResult.index;
        removalPage = findResult.page;

        WriteToJITRegionScope writeScope;
        ValidateNonReentrancyScope validateNonReentrancy(metadata());

        clear(entry);
        removalPage->clearIsInUse(removalIndex);
    }

    // This implementation allows recursive uses of the MacroAssembler, forming a stack
    // like data structure. The in use pin (top of the stack) will always be at the lowest
    // page and lowest index in the bit vector. So when we deallocate the current top of the
    // stack, we need to find the next entry in the stack, and ensure it's at the lowest index.
    // So if there are other entries for this current thread, we find the one with the highest
    // pin (next top of stack value), and replace it at the index we were using before.
    // Allocation also maintains this invariant by always placing the newest entry for
    // the current thread at the lowest index.
    {
        uint64_t key = keyForCurrentThread();
        bool found = false;
        uint64_t maxPin = 0;
        size_t maxIndex = std::numeric_limits<size_t>::max();
        Page* maxPage = nullptr;

        forEachEntry([&] (Page& page, Entry& entry, size_t index) {
            RELEASE_ASSERT(entry.pin);
            if (entry.key == key && entry.pin > maxPin) {
                found = true;
                maxPin = entry.pin;
                maxIndex = index;
                maxPage = &page;
            }

            return IterationStatus::Continue;
        });

        if (found) {
            RELEASE_ASSERT(removalIndex < maxIndex);

            WriteToJITRegionScope writeScope;
            ValidateNonReentrancyScope validateNonReentrancy(metadata());

            removalPage->entry(removalIndex) = maxPage->entry(maxIndex);
            clear(maxPage->entry(maxIndex));
            maxPage->clearIsInUse(maxIndex);
            removalPage->setIsInUse(removalIndex);
        }
    }
}

#endif // CPU(ARM64E) && ENABLE(JIT)

} // namespace JSC
