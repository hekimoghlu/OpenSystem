/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 28, 2023.
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

#include "IsoHeapImpl.h"
#include "IsoTLS.h"
#include "bmalloc.h"

namespace bmalloc {

template<typename Type>
void* IsoTLS::allocate(api::IsoHeap<Type>& handle, bool abortOnFailure)
{
    return allocateImpl<typename api::IsoHeap<Type>::Config>(handle, abortOnFailure);
}

template<typename Type>
void IsoTLS::deallocate(api::IsoHeap<Type>& handle, void* p)
{
    if (!p)
        return;
    deallocateImpl<typename api::IsoHeap<Type>::Config>(handle, p);
}

template<typename Type>
void IsoTLS::scavenge(api::IsoHeap<Type>& handle)
{
    IsoTLS* tls = get();
    if (!tls)
        return;
    if (!handle.isInitialized())
        return;
    unsigned offset = handle.allocatorOffset();
    if (offset < tls->m_extent)
        reinterpret_cast<IsoAllocator<typename api::IsoHeap<Type>::Config>*>(tls->m_data + offset)->scavenge();
    offset = handle.deallocatorOffset();
    if (offset < tls->m_extent)
        reinterpret_cast<IsoDeallocator<typename api::IsoHeap<Type>::Config>*>(tls->m_data + offset)->scavenge();
    handle.impl().scavengeNow();
}

template<typename Config, typename Type>
void* IsoTLS::allocateImpl(api::IsoHeap<Type>& handle, bool abortOnFailure)
{
    unsigned offset = handle.allocatorOffset();
    IsoTLS* tls = get();
    if (!tls || offset >= tls->m_extent)
        return allocateSlow<Config>(handle, abortOnFailure);
    return tls->allocateFast<Config>(offset, abortOnFailure);
}

template<typename Config>
void* IsoTLS::allocateFast(unsigned offset, bool abortOnFailure)
{
    return reinterpret_cast<IsoAllocator<Config>*>(m_data + offset)->allocate(abortOnFailure);
}

template<typename Config, typename Type>
BNO_INLINE void* IsoTLS::allocateSlow(api::IsoHeap<Type>& handle, bool abortOnFailure)
{
    for (;;) {
        switch (s_mallocFallbackState) {
        case MallocFallbackState::Undecided:
            determineMallocFallbackState();
            continue;
        case MallocFallbackState::FallBackToMalloc:
            return api::tryMalloc(Config::objectSize);
        case MallocFallbackState::DoNotFallBack:
            break;
        }
        break;
    }
    
    auto debugMallocResult = debugMalloc(Config::objectSize);
    if (debugMallocResult.usingDebugHeap)
        return debugMallocResult.ptr;
    
    IsoTLS* tls = ensureHeapAndEntries(handle);
    
    return tls->allocateFast<Config>(handle.allocatorOffset(), abortOnFailure);
}

template<typename Config, typename Type>
void IsoTLS::deallocateImpl(api::IsoHeap<Type>& handle, void* p)
{
    unsigned offset = handle.deallocatorOffset();
    IsoTLS* tls = get();
    // Note that this bounds check would be here even if we didn't have to support DebugHeap,
    // since we don't want unpredictable behavior if offset or m_extent ever got corrupted.
    if (!tls || offset >= tls->m_extent)
        deallocateSlow<Config>(handle, p);
    else
        tls->deallocateFast<Config>(offset, p);
}

template<typename Config>
void IsoTLS::deallocateFast(unsigned offset, void* p)
{
    reinterpret_cast<IsoDeallocator<Config>*>(m_data + offset)->deallocate(p);
}

template<typename Config, typename Type>
BNO_INLINE void IsoTLS::deallocateSlow(api::IsoHeap<Type>& handle, void* p)
{
    for (;;) {
        switch (s_mallocFallbackState) {
        case MallocFallbackState::Undecided:
            determineMallocFallbackState();
            continue;
        case MallocFallbackState::FallBackToMalloc:
            return api::free(p);
        case MallocFallbackState::DoNotFallBack:
            break;
        }
        break;
    }
    
    if (debugFree(p))
        return;
    
    RELEASE_BASSERT(handle.isInitialized());
    
    IsoTLS* tls = ensureEntries(std::max(handle.allocatorOffset(), handle.deallocatorOffset()));
    
    tls->deallocateFast<Config>(handle.deallocatorOffset(), p);
}

inline IsoTLS* IsoTLS::get()
{
#if HAVE_PTHREAD_MACHDEP_H
    return static_cast<IsoTLS*>(_pthread_getspecific_direct(tlsKey));
#else
    if (!s_didInitialize)
        return nullptr;
    return static_cast<IsoTLS*>(pthread_getspecific(s_tlsKey));
#endif
}

inline void IsoTLS::set(IsoTLS* tls)
{
#if HAVE_PTHREAD_MACHDEP_H
    _pthread_setspecific_direct(tlsKey, tls);
#else
    pthread_setspecific(s_tlsKey, tls);
#endif
}

template<typename Type>
void IsoTLS::ensureHeap(api::IsoHeap<Type>& handle)
{
    if (!handle.isInitialized()) {
        std::lock_guard<Mutex> locker(handle.m_initializationLock);
        if (!handle.isInitialized()) {
            auto* heap = new IsoHeapImpl<typename api::IsoHeap<Type>::Config>();
            std::atomic_thread_fence(std::memory_order_seq_cst);
            handle.setAllocatorOffset(heap->allocatorOffset());
            handle.setDeallocatorOffset(heap->deallocatorOffset());
            handle.m_impl = heap;
        }
    }
}

template<typename Type>
BNO_INLINE IsoTLS* IsoTLS::ensureHeapAndEntries(api::IsoHeap<Type>& handle)
{
    RELEASE_BASSERT(
        !get()
        || handle.allocatorOffset() >= get()->m_extent
        || handle.deallocatorOffset() >= get()->m_extent);
    ensureHeap(handle);
    return ensureEntries(std::max(handle.allocatorOffset(), handle.deallocatorOffset()));
}

} // namespace bmalloc

