/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 2, 2022.
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

#include "CompactAllocationMode.h"
#include "IsoConfig.h"
#include "Mutex.h"

#if BUSE(LIBPAS)
#include "bmalloc_heap_ref.h"
#endif

#if BENABLE_MALLOC_HEAP_BREAKDOWN
#include <malloc/malloc.h>
#endif

namespace bmalloc {

template<typename Config> class IsoHeapImpl;

namespace api {

// You have to declare IsoHeaps this way:
//
// static IsoHeap<type> myTypeHeap;
//
// It's not valid to create an IsoHeap except in static storage.

#if BUSE(LIBPAS)
BEXPORT void* isoAllocate(pas_heap_ref&);
BEXPORT void* isoTryAllocate(pas_heap_ref&);
BEXPORT void* isoAllocateCompact(pas_heap_ref&);
BEXPORT void* isoTryAllocateCompact(pas_heap_ref&);
BEXPORT void isoDeallocate(void* ptr);

// The name "LibPasBmallocHeapType" is important for the pas_status_reporter to work right.
template<typename LibPasBmallocHeapType>
struct IsoHeapBase {
    constexpr IsoHeapBase(const char* = nullptr) { }

    void scavenge() { }
    void initialize() { }

    bool isInitialized()
    {
        return true;
    }

    static pas_heap_ref& provideHeap()
    {
        static const bmalloc_type type = BMALLOC_TYPE_INITIALIZER(sizeof(LibPasBmallocHeapType), alignof(LibPasBmallocHeapType), __PRETTY_FUNCTION__);
        static pas_heap_ref heap = BMALLOC_HEAP_REF_INITIALIZER(&type);
        return heap;
    }
};

template<typename LibPasBmallocHeapType>
struct IsoHeap : public IsoHeapBase<LibPasBmallocHeapType> {
    using IsoHeapBase<LibPasBmallocHeapType>::provideHeap;

    constexpr IsoHeap(const char* name = nullptr): IsoHeapBase<LibPasBmallocHeapType>(name) { }
    
    void* allocate()
    {
        return isoAllocate(provideHeap());
    }
    
    void* tryAllocate()
    {
        return isoTryAllocate(provideHeap());
    }
    
    void deallocate(void* p)
    {
        isoDeallocate(p);
    }
};

template<typename LibPasBmallocHeapType>
struct CompactIsoHeap : public IsoHeapBase<LibPasBmallocHeapType> {
    using IsoHeapBase<LibPasBmallocHeapType>::provideHeap;

    constexpr CompactIsoHeap(const char* name = nullptr): IsoHeapBase<LibPasBmallocHeapType>(name) { }

    void* allocate()
    {
        return isoAllocateCompact(provideHeap());
    }

    void* tryAllocate()
    {
        return isoTryAllocateCompact(provideHeap());
    }

    void deallocate(void* p)
    {
        isoDeallocate(p);
    }
};

#else // BUSE(LIBPAS) -> so !BUSE(LIBPAS)
template<typename Type>
struct IsoHeapBase {
    typedef IsoConfig<sizeof(Type)> Config;

#if BENABLE_MALLOC_HEAP_BREAKDOWN
    IsoHeapBase(const char* = nullptr);
#else
    constexpr IsoHeapBase(const char* = nullptr) { }
#endif

    void* allocate();
    void* tryAllocate();
    void deallocate(void* p);
    
    void scavenge();
    
    void initialize();
    bool isInitialized();
    
    unsigned allocatorOffset() { return m_allocatorOffsetPlusOne - 1; }
    void setAllocatorOffset(unsigned value) { m_allocatorOffsetPlusOne = value + 1; }
    
    unsigned deallocatorOffset() { return m_deallocatorOffsetPlusOne - 1; }
    void setDeallocatorOffset(unsigned value) { m_deallocatorOffsetPlusOne = value + 1; }
    
    IsoHeapImpl<Config>& impl();
    
    Mutex m_initializationLock;
    unsigned m_allocatorOffsetPlusOne { 0 };
    unsigned m_deallocatorOffsetPlusOne { 0 };
    IsoHeapImpl<Config>* m_impl { nullptr };

#if BENABLE_MALLOC_HEAP_BREAKDOWN
    malloc_zone_t* m_zone;
#endif
};

template<typename Type>
struct IsoHeap : public IsoHeapBase<Type> {
    constexpr IsoHeap(const char* name = nullptr): IsoHeapBase<Type>(name) { }
};

template<typename Type>
struct CompactIsoHeap : public IsoHeapBase<Type> {
    constexpr CompactIsoHeap(const char* name = nullptr): IsoHeapBase<Type>(name) { }
};
#endif // BUSE(LIBPAS) -> so end of !BUSE(LIBPAS)

// Use this together with MAKE_BISO_MALLOCED_IMPL.
#define MAKE_BISO_MALLOCED(isoType, heapType, exportMacro) \
public: \
    static exportMacro ::bmalloc::api::heapType<isoType>& bisoHeap(); \
    \
    void* operator new(size_t, void* p) { return p; } \
    void* operator new[](size_t, void* p) { return p; } \
    \
    exportMacro void* operator new(size_t size);\
    exportMacro void operator delete(void* p);\
    \
    void* operator new[](size_t size) = delete; \
    void operator delete[](void* p) = delete; \
    \
    void* operator new(size_t, NotNullTag, void* location) \
    { \
        ASSERT(location); \
        return location; \
    } \
    \
    exportMacro static void freeAfterDestruction(void*); \
    \
    using WTFIsFastAllocated = int; \
private: \
    using __makeBisoMallocedMacroSemicolonifier BUNUSED_TYPE_ALIAS = int

// Use this together with MAKE_BISO_MALLOCED_IMPL.
#define MAKE_BISO_MALLOCED_COMPACT(isoType, heapType, exportMacro) \
public: \
    static exportMacro ::bmalloc::api::heapType<isoType>& bisoHeap(); \
    \
    void* operator new(size_t, void* p) { return p; } \
    void* operator new[](size_t, void* p) { return p; } \
    \
    exportMacro void* operator new(size_t size);\
    exportMacro void operator delete(void* p);\
    \
    void* operator new[](size_t size) = delete; \
    void operator delete[](void* p) = delete; \
    \
    void* operator new(size_t, NotNullTag, void* location) \
    { \
        ASSERT(location); \
        return location; \
    } \
    \
    exportMacro static void freeAfterDestruction(void*); \
    \
    using WTFIsFastAllocated = int; \
private: \
    using __makeBisoMallocedMacroSemicolonifier BUNUSED_TYPE_ALIAS = int

// Use this together with MAKE_BISO_MALLOCED for template classes.
#define MAKE_BISO_MALLOCED_TEMPLATE_IMPL(templateParameters, isoType) \
templateParameters \
::bmalloc::api::IsoHeap<isoType>& isoType::bisoHeap() \
{ \
    static ::bmalloc::api::IsoHeap<isoType> heap("WebKit_"#isoType); \
    return heap; \
} \
\
templateParameters \
void* isoType::operator new(size_t size) \
{ \
    RELEASE_BASSERT(size == sizeof(isoType)); \
    return bisoHeap().allocate(); \
} \
\
templateParameters \
void isoType::operator delete(void* p) \
{ \
    bisoHeap().deallocate(p); \
} \
\
templateParameters \
void isoType::freeAfterDestruction(void* p) \
{ \
    bisoHeap().deallocate(p); \
} \
\
using __makeBisoMallocedMacroSemicolonifier BUNUSED_TYPE_ALIAS = int

} } // namespace bmalloc::api

#endif // !BUSE(TZONE)
