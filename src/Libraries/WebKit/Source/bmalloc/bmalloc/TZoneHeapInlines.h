/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 1, 2023.
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

#if BUSE(TZONE)

#include "TZoneHeap.h"

namespace bmalloc { namespace api {

// This is most appropriate for classes defined in a .cpp/.mm file.

#define MAKE_BTZONE_MALLOCED_COMMON_INLINE(_type, _compactMode, _exportMacro) \
public: \
    using HeapRef = ::bmalloc::api::HeapRef; \
    using SizeAndAlignment = ::bmalloc::api::SizeAndAlignment; \
    using TZoneMallocFallback = ::bmalloc::api::TZoneMallocFallback; \
    \
    BINLINE void* operator new(size_t, void* p) { return p; } \
    BINLINE void* operator new[](size_t, void* p) { return p; } \
    \
    void* operator new[](size_t size) = delete; \
    void operator delete[](void* p) = delete; \
    \
    BINLINE void* operator new(size_t, NotNullTag, void* location) \
    { \
        ASSERT(location); \
        return location; \
    } \
    \
    void* operator new(size_t size) \
    { \
        static HeapRef s_heapRef; \
        static const TZoneSpecification s_heapSpec = { &s_heapRef, sizeof(_type), SizeAndAlignment::encode<_type>() TZONE_SPEC_NAME_ARG(#_type) }; \
    \
        if (BUNLIKELY(!s_heapRef || size != sizeof(_type))) \
            return ::bmalloc::api::tzoneAllocate ## _compactMode ## Slow(size, s_heapSpec); \
        BASSERT(::bmalloc::api::tzoneMallocFallback > TZoneMallocFallback::ForceDebugMalloc); \
        return ::bmalloc::api::tzoneAllocate ## _compactMode(s_heapRef); \
    } \
    \
    BINLINE void operator delete(void* p) \
    { \
        ::bmalloc::api::tzoneFree(p); \
    } \
    \
    BINLINE static void freeAfterDestruction(void* p) \
    { \
        ::bmalloc::api::tzoneFree(p); \
    } \
    \
    using WTFIsFastAllocated = int;

#define MAKE_BTZONE_MALLOCED_INLINE(_type, _compactMode) \
public: \
    MAKE_BTZONE_MALLOCED_COMMON_INLINE(_type, _compactMode, BNOEXPORT) \
private: \
    using __makeBtzoneMallocedInlineMacroSemicolonifier BUNUSED_TYPE_ALIAS = int


#define MAKE_BTZONE_MALLOCED_IMPL(_type, _compactMode) \
::bmalloc::api::HeapRef _type::s_heapRef; \
\
const TZoneSpecification _type::s_heapSpec = { &_type::s_heapRef, sizeof(_type), SizeAndAlignment::encode<_type>() TZONE_SPEC_NAME_ARG(#_type) }; \
\
void* _type::operatorNewSlow(size_t size) \
{ \
    return ::bmalloc::api::tzoneAllocate ## _compactMode ## Slow(size, s_heapSpec); \
} \
\
using __makeBtzoneMallocedInlineMacroSemicolonifier BUNUSED_TYPE_ALIAS = int

} } // namespace bmalloc::api

#endif // BUSE(TZONE)
