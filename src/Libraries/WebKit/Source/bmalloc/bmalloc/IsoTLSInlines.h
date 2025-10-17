/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 16, 2024.
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

#if !BUSE(TZONE)

#include "Environment.h"
#include "IsoHeapImpl.h"
#include "IsoMallocFallback.h"
#include "IsoTLS.h"
#include "bmalloc.h"

#if !BUSE(LIBPAS)

#if BOS(DARWIN)
#include <malloc/malloc.h>
#endif

namespace bmalloc {

template<typename Type>
void* IsoTLS::allocate(api::IsoHeapBase<Type>& handle, bool abortOnFailure)
{
    return allocateImpl<typename api::IsoHeapBase<Type>::Config>(handle, abortOnFailure);
}

template<typename Type>
void IsoTLS::deallocate(api::IsoHeapBase<Type>& handle, void* p)
{
    if (!p)
        return;
    deallocateImpl<typename api::IsoHeapBase<Type>::Config>(handle, p);
}

template<typename Type>
void IsoTLS::scavenge(api::IsoHeapBase<Type>& handle)
{
    IsoTLS* tls = get();
    if (!tls)
        return;
    if (!handle.isInitialized())
        return;
    unsigned offset = handle.allocatorOffset();
    if (offset < tls->m_extent)
        reinterpret_cast<IsoAllocator<typename api::IsoHeapBase<Type>::Config>*>(tls->m_data + offset)->scavenge(handle.impl());
    offset = handle.deallocatorOffset();
    if (offset < tls->m_extent)
        reinterpret_cast<IsoDeallocator<typename api::IsoHeapBase<Type>::Config>*>(tls->m_data + offset)->scavenge();
    handle.impl().scavengeNow();
}

template<typename Config, typename Type>
void* IsoTLS::allocateImpl(api::IsoHeapBase<Type>& handle, bool abortOnFailure)
{
    unsigned offset = handle.allocatorOffset();
    IsoTLS* tls = get();
    if (!tls || offset >= tls->m_extent)
        return allocateSlow<Config>(handle, abortOnFailure);
    return tls->allocateFast<Config>(handle, offset, abortOnFailure);
}

template<typename Config, typename Type>
void* IsoTLS::allocateFast(api::IsoHeapBase<Type>& handle, unsigned offset, bool abortOnFailure)
{
    return reinterpret_cast<IsoAllocator<Config>*>(m_data + offset)->allocate(handle.impl(), abortOnFailure);
}

template<typename Config, typename Type>
BNO_INLINE void* IsoTLS::allocateSlow(api::IsoHeapBase<Type>& handle, bool abortOnFailure)
{
    IsoMallocFallback::MallocResult fallbackResult = IsoMallocFallback::tryMalloc(
        Config::objectSize,
        CompactAllocationMode::NonCompact
#if BENABLE_MALLOC_HEAP_BREAKDOWN
        , handle.m_zone
#endif
        );
    if (fallbackResult.didFallBack)
        return fallbackResult.ptr;
    
    // If debug heap is enabled, IsoMallocFallback::mallocFallbackState becomes MallocFallbackState::FallBackToMalloc.
    BASSERT(!Environment::get()->isDebugHeapEnabled());
    
    IsoTLS* tls = ensureHeapAndEntries(handle);
    
    return tls->allocateFast<Config>(handle, handle.allocatorOffset(), abortOnFailure);
}

template<typename Config, typename Type>
void IsoTLS::deallocateImpl(api::IsoHeapBase<Type>& handle, void* p)
{
    unsigned offset = handle.deallocatorOffset();
    IsoTLS* tls = get();
    // Note that this bounds check would be here even if we didn't have to support DebugHeap,
    // since we don't want unpredictable behavior if offset or m_extent ever got corrupted.
    if (!tls || offset >= tls->m_extent)
        deallocateSlow<Config>(handle, p);
    else
        tls->deallocateFast<Config>(handle, offset, p);
}

template<typename Config, typename Type>
void IsoTLS::deallocateFast(api::IsoHeapBase<Type>& handle, unsigned offset, void* p)
{
    reinterpret_cast<IsoDeallocator<Config>*>(m_data + offset)->deallocate(handle, p);
}

template<typename Config, typename Type>
BNO_INLINE void IsoTLS::deallocateSlow(api::IsoHeapBase<Type>& handle, void* p)
{
    bool result = IsoMallocFallback::tryFree(
        p
#if BENABLE_MALLOC_HEAP_BREAKDOWN
        , handle.m_zone
#endif
        );
    if (result)
        return;
    
    // If debug heap is enabled, IsoMallocFallback::mallocFallbackState becomes MallocFallbackState::FallBackToMalloc.
    BASSERT(!Environment::get()->isDebugHeapEnabled());
    
    RELEASE_BASSERT(handle.isInitialized());
    
    IsoTLS* tls = ensureEntries(std::max(handle.allocatorOffset(), handle.deallocatorOffset()));
    
    tls->deallocateFast<Config>(handle, handle.deallocatorOffset(), p);
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
void IsoTLS::ensureHeap(api::IsoHeapBase<Type>& handle)
{
    if (!handle.isInitialized()) {
        LockHolder locker(handle.m_initializationLock);
        if (!handle.isInitialized())
            handle.initialize();
    }
}

template<typename Type>
BNO_INLINE IsoTLS* IsoTLS::ensureHeapAndEntries(api::IsoHeapBase<Type>& handle)
{
    RELEASE_BASSERT(
        !get()
        || handle.allocatorOffset() >= get()->m_extent
        || handle.deallocatorOffset() >= get()->m_extent);
    ensureHeap(handle);
    return ensureEntries(std::max(handle.allocatorOffset(), handle.deallocatorOffset()));
}

} // namespace bmalloc

#endif
#endif // !BUSE(TZONE)
