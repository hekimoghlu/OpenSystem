/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 4, 2025.
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
#include "IsoTLS.h"

#include "DebugHeap.h"
#include "Environment.h"
#include "Gigacage.h"
#include "IsoTLSEntryInlines.h"
#include "IsoTLSInlines.h"
#include "IsoTLSLayout.h"

#include <stdio.h>

namespace bmalloc {

IsoTLS::MallocFallbackState IsoTLS::s_mallocFallbackState;

#if !HAVE_PTHREAD_MACHDEP_H
bool IsoTLS::s_didInitialize;
pthread_key_t IsoTLS::s_tlsKey;
#endif

void IsoTLS::scavenge()
{
    if (IsoTLS* tls = get()) {
        tls->forEachEntry(
            [&] (IsoTLSEntry* entry, void* data) {
                entry->scavenge(data);
            });
    }
}

IsoTLS::IsoTLS()
{
}

IsoTLS* IsoTLS::ensureEntries(unsigned offset)
{
    RELEASE_BASSERT(!get() || offset >= get()->m_extent);
    
    static std::once_flag onceFlag;
    std::call_once(
        onceFlag,
        [] () {
            setvbuf(stderr, NULL, _IONBF, 0);
#if HAVE_PTHREAD_MACHDEP_H
            pthread_key_init_np(tlsKey, destructor);
#else
            int error = pthread_key_create(&s_tlsKey, destructor);
            if (error)
                BCRASH();
            s_didInitialize = true;
#endif
        });
    
    IsoTLS* tls = get();
    IsoTLSLayout& layout = *PerProcess<IsoTLSLayout>::get();

    IsoTLSEntry* oldLastEntry = tls ? tls->m_lastEntry : nullptr;
    RELEASE_BASSERT(!oldLastEntry || oldLastEntry->offset() < offset);
    
    IsoTLSEntry* startEntry = oldLastEntry ? oldLastEntry : layout.head();
    
    IsoTLSEntry* targetEntry = startEntry;
    size_t requiredCapacity = 0;
    if (startEntry) {
        for (;;) {
            RELEASE_BASSERT(targetEntry);
            RELEASE_BASSERT(targetEntry->offset() <= offset);
            if (targetEntry->offset() == offset)
                break;
            targetEntry = targetEntry->m_next;
        }
        RELEASE_BASSERT(targetEntry);
        requiredCapacity = targetEntry->extent();
    }
    
    if (!tls || requiredCapacity > tls->m_capacity) {
        size_t requiredSize = sizeForCapacity(requiredCapacity);
        size_t goodSize = roundUpToMultipleOf(vmPageSize(), requiredSize);
        size_t goodCapacity = capacityForSize(goodSize);
        void* memory = vmAllocate(goodSize);
        IsoTLS* newTLS = new (memory) IsoTLS();
        newTLS->m_capacity = goodCapacity;
        if (tls) {
            RELEASE_BASSERT(oldLastEntry);
            RELEASE_BASSERT(layout.head());
            layout.head()->walkUpToInclusive(
                oldLastEntry,
                [&] (IsoTLSEntry* entry) {
                    void* src = tls->m_data + entry->offset();
                    void* dst = newTLS->m_data + entry->offset();
                    entry->move(src, dst);
                    entry->destruct(src);
                });
            vmDeallocate(tls, tls->size());
        }
        tls = newTLS;
        set(tls);
    }
    
    if (startEntry) {
        startEntry->walkUpToInclusive(
            targetEntry,
            [&] (IsoTLSEntry* entry) {
                entry->construct(tls->m_data + entry->offset());
            });
        
        tls->m_lastEntry = targetEntry;
        tls->m_extent = targetEntry->extent();
    }
    
    return tls;
}

void IsoTLS::destructor(void* arg)
{
    IsoTLS* tls = static_cast<IsoTLS*>(arg);
    RELEASE_BASSERT(tls);
    tls->forEachEntry(
        [&] (IsoTLSEntry* entry, void* data) {
            entry->scavenge(data);
            entry->destruct(data);
        });
}

size_t IsoTLS::sizeForCapacity(unsigned capacity)
{
    return BOFFSETOF(IsoTLS, m_data) + capacity;
}

unsigned IsoTLS::capacityForSize(size_t size)
{
    return size - sizeForCapacity(0);
}

size_t IsoTLS::size()
{
    return sizeForCapacity(m_capacity);
}

template<typename Func>
void IsoTLS::forEachEntry(const Func& func)
{
    if (!m_lastEntry)
        return;
    PerProcess<IsoTLSLayout>::get()->head()->walkUpToInclusive(
        m_lastEntry,
        [&] (IsoTLSEntry* entry) {
            func(entry, m_data + entry->offset());
        });
}

bool IsoTLS::isUsingDebugHeap()
{
    return PerProcess<Environment>::get()->isDebugHeapEnabled();
}

auto IsoTLS::debugMalloc(size_t size) -> DebugMallocResult
{
    DebugMallocResult result;
    if ((result.usingDebugHeap = isUsingDebugHeap()))
        result.ptr = PerProcess<DebugHeap>::get()->malloc(size);
    return result;
}

bool IsoTLS::debugFree(void* p)
{
    if (isUsingDebugHeap()) {
        PerProcess<DebugHeap>::get()->free(p);
        return true;
    }
    return false;
}

void IsoTLS::determineMallocFallbackState()
{
    static std::once_flag onceFlag;
    std::call_once(
        onceFlag,
        [] {
            if (s_mallocFallbackState != MallocFallbackState::Undecided)
                return;

#if GIGACAGE_ENABLED
            if (!Gigacage::shouldBeEnabled()) {
                s_mallocFallbackState = MallocFallbackState::FallBackToMalloc;
                return;
            }
            const char* env = getenv("bmalloc_IsoHeap");
            if (env && (!strcasecmp(env, "false") || !strcasecmp(env, "no") || !strcmp(env, "0")))
                s_mallocFallbackState = MallocFallbackState::FallBackToMalloc;
            else
                s_mallocFallbackState = MallocFallbackState::DoNotFallBack;
#else
            s_mallocFallbackState = MallocFallbackState::FallBackToMalloc;
#endif
        });
}

} // namespace bmalloc

