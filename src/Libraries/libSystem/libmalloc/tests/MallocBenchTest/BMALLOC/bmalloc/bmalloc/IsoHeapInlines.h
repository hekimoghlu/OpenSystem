/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 27, 2022.
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

#include "DeferredDecommitInlines.h"
#include "DeferredTriggerInlines.h"
#include "EligibilityResultInlines.h"
#include "FreeListInlines.h"
#include "IsoAllocatorInlines.h"
#include "IsoDeallocatorInlines.h"
#include "IsoDirectoryInlines.h"
#include "IsoDirectoryPageInlines.h"
#include "IsoHeapImplInlines.h"
#include "IsoHeap.h"
#include "IsoPageInlines.h"
#include "IsoTLSAllocatorEntryInlines.h"
#include "IsoTLSDeallocatorEntryInlines.h"
#include "IsoTLSEntryInlines.h"
#include "IsoTLSInlines.h"

namespace bmalloc { namespace api {

template<typename Type>
void* IsoHeap<Type>::allocate()
{
    bool abortOnFailure = true;
    return IsoTLS::allocate(*this, abortOnFailure);
}

template<typename Type>
void* IsoHeap<Type>::tryAllocate()
{
    bool abortOnFailure = false;
    return IsoTLS::allocate(*this, abortOnFailure);
}

template<typename Type>
void IsoHeap<Type>::deallocate(void* p)
{
    IsoTLS::deallocate(*this, p);
}

template<typename Type>
void IsoHeap<Type>::scavenge()
{
    IsoTLS::scavenge(*this);
}

template<typename Type>
bool IsoHeap<Type>::isInitialized()
{
    std::atomic<unsigned>* atomic =
        reinterpret_cast<std::atomic<unsigned>*>(&m_allocatorOffsetPlusOne);
    return !!atomic->load(std::memory_order_acquire);
}

template<typename Type>
auto IsoHeap<Type>::impl() -> IsoHeapImpl<Config>&
{
    IsoTLS::ensureHeap(*this);
    return *m_impl;
}

// This is most appropraite for template classes.
#define MAKE_BISO_MALLOCED_INLINE(isoType) \
public: \
    static ::bmalloc::api::IsoHeap<isoType>& bisoHeap() \
    { \
        static ::bmalloc::api::IsoHeap<isoType> heap; \
        return heap; \
    } \
    \
    void* operator new(size_t, void* p) { return p; } \
    void* operator new[](size_t, void* p) { return p; } \
    \
    void* operator new(size_t size) \
    { \
        RELEASE_BASSERT(size == sizeof(isoType)); \
        return bisoHeap().allocate(); \
    } \
    \
    void operator delete(void* p) \
    { \
        bisoHeap().deallocate(p); \
    } \
    \
    void* operator new[](size_t size) = delete; \
    void operator delete[](void* p) = delete; \
private: \
typedef int __makeBisoMallocedInlineMacroSemicolonifier

#define MAKE_BISO_MALLOCED_IMPL(isoType) \
::bmalloc::api::IsoHeap<isoType>& isoType::bisoHeap() \
{ \
    static ::bmalloc::api::IsoHeap<isoType> heap; \
    return heap; \
} \
\
void* isoType::operator new(size_t size) \
{ \
    RELEASE_BASSERT(size == sizeof(isoType)); \
    return bisoHeap().allocate(); \
} \
\
void isoType::operator delete(void* p) \
{ \
    bisoHeap().deallocate(p); \
} \
\
struct MakeBisoMallocedImplMacroSemicolonifier##isoType { }

} } // namespace bmalloc::api

