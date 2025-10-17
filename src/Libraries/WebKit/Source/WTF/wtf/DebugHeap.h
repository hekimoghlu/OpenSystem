/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 23, 2021.
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

#include <wtf/ExportMacros.h>
#include <wtf/Platform.h>

#if ENABLE(MALLOC_HEAP_BREAKDOWN)
#include <mutex>
#if OS(DARWIN)
#include <malloc/malloc.h>
#endif
#endif

namespace WTF {

#define DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(Type) DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER_AND_EXPORT(Type, WTF_EXPORT_PRIVATE)
#define DECLARE_COMPACT_ALLOCATOR_WITH_HEAP_IDENTIFIER(Type) DECLARE_COMPACT_ALLOCATOR_WITH_HEAP_IDENTIFIER_AND_EXPORT(Type, WTF_EXPORT_PRIVATE)

#if ENABLE(MALLOC_HEAP_BREAKDOWN)

class DebugHeap {
public:
    WTF_EXPORT_PRIVATE DebugHeap(const char* heapName);

    WTF_EXPORT_PRIVATE void* malloc(size_t);
    WTF_EXPORT_PRIVATE void* calloc(size_t numElements, size_t elementSize);
    WTF_EXPORT_PRIVATE void* memalign(size_t alignment, size_t, bool crashOnFailure);
    WTF_EXPORT_PRIVATE void* realloc(void*, size_t);
    WTF_EXPORT_PRIVATE void free(void*);

private:
#if OS(DARWIN)
    malloc_zone_t* m_zone;
#endif
};

#define DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER_AND_EXPORT(Type, Export) \
    struct Type##Malloc { \
        static Export WTF::DebugHeap& debugHeap(); \
\
        static void* malloc(size_t size) { return debugHeap().malloc(size); } \
\
        static void* tryMalloc(size_t size) { return debugHeap().malloc(size); } \
\
        static void* zeroedMalloc(size_t size) { return debugHeap().calloc(1, size); } \
\
        static void* tryZeroedMalloc(size_t size) { return debugHeap().calloc(1, size); } \
\
        static void* realloc(void* p, size_t size) { return debugHeap().realloc(p, size); } \
\
        static void* tryRealloc(void* p, size_t size) { return debugHeap().realloc(p, size); } \
\
        static void free(void* p) { debugHeap().free(p); } \
\
        static constexpr ALWAYS_INLINE size_t nextCapacity(size_t capacity) { return capacity + capacity / 4 + 1; } \
    }

#define DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(Type) \
    WTF::DebugHeap& Type##Malloc::debugHeap() \
    { \
        static LazyNeverDestroyed<WTF::DebugHeap> heap; \
        static std::once_flag onceKey; \
        std::call_once(onceKey, [&] { \
            heap.construct(#Type); \
        }); \
        return heap; \
    } \
    struct MakeDebugHeapMallocedImplMacroSemicolonifier##Type { }

#define DECLARE_COMPACT_ALLOCATOR_WITH_HEAP_IDENTIFIER_AND_EXPORT(Type, Export) \
    DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER_AND_EXPORT(Type, Export)

#define DEFINE_COMPACT_ALLOCATOR_WITH_HEAP_IDENTIFIER(Type) \
    DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(Type)

#else // ENABLE(MALLOC_HEAP_BREAKDOWN)

#define DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER_AND_EXPORT(Type, Export) \
    using Type##Malloc = FastMalloc

#define DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(Type) \
    struct MakeDebugHeapMallocedImplMacroSemicolonifier##Type { }

#define DECLARE_COMPACT_ALLOCATOR_WITH_HEAP_IDENTIFIER_AND_EXPORT(Type, Export) \
    using Type##Malloc = FastCompactMalloc

#define DEFINE_COMPACT_ALLOCATOR_WITH_HEAP_IDENTIFIER(Type) \
    struct MakeDebugHeapMallocedImplMacroSemicolonifier##Type { }

#endif

} // namespace WTF
