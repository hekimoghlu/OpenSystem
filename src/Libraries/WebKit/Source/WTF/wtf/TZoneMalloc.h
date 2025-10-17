/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 25, 2021.
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

#include <wtf/ForbidHeapAllocation.h>
#include <wtf/Platform.h>

#define WTF_NOEXPORT

#if USE(SYSTEM_MALLOC) || !USE(TZONE_MALLOC)

#include <wtf/FastMalloc.h>

// class allocators with FastMalloc fallback if TZoneHeap is disabled.
#define WTF_MAKE_TZONE_ALLOCATED(name) WTF_MAKE_FAST_ALLOCATED
#define WTF_MAKE_TZONE_ALLOCATED_EXPORT(name, exportMacro) WTF_MAKE_FAST_ALLOCATED

// struct allocators with FastMalloc fallback if TZoneHeap is disabled.
#define WTF_MAKE_STRUCT_TZONE_ALLOCATED(name) WTF_MAKE_STRUCT_FAST_ALLOCATED

// template allocators with FastMalloc fallback if TZoneHeap is disabled.
#define WTF_MAKE_TZONE_ALLOCATED_TEMPLATE(name) WTF_MAKE_FAST_ALLOCATED
#define WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_EXPORT(name, exportMacro) WTF_MAKE_FAST_ALLOCATED

// special class (e.g. those used with CompactPtr) allocators with FastMalloc fallback if TZoneHeap is disabled.
#define WTF_MAKE_COMPACT_TZONE_ALLOCATED(name) WTF_MAKE_FAST_COMPACT_ALLOCATED
#define WTF_MAKE_COMPACT_TZONE_ALLOCATED_EXPORT(name, exportMacro) WTF_MAKE_FAST_COMPACT_ALLOCATED

#if USE(SYSTEM_MALLOC) || !USE(ISO_MALLOC)

// class allocators with IsoHeap fallback if TZoneHeap and IsoHeap are disabled.
#define WTF_MAKE_TZONE_OR_ISO_ALLOCATED(name) WTF_MAKE_FAST_ALLOCATED
#define WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(name, exportMacro) WTF_MAKE_FAST_ALLOCATED

// template allocators with IsoHeap fallback if TZoneHeap and IsoHeap are disabled.
#define WTF_MAKE_TZONE_OR_ISO_ALLOCATED_TEMPLATE(name) WTF_MAKE_FAST_ALLOCATED

// class allocators with IsoHeap fallback if TZoneHeap and IsoHeap are disabled.
#define WTF_MAKE_COMPACT_TZONE_OR_ISO_ALLOCATED(name) WTF_MAKE_FAST_COMPACT_ALLOCATED
#define WTF_MAKE_COMPACT_TZONE_OR_ISO_ALLOCATED_EXPORT(name, exportMacro) WTF_MAKE_FAST_COMPACT_ALLOCATED

// template implementation to go with WTF_MAKE_TZONE_OR_ISO_ALLOCATED_TEMPLATE
// if TZoneHeap and IsoHeap are disabled. This should be added immediately after the
// template definition.
#define WTF_MAKE_TZONE_OR_ISO_ALLOCATED_TEMPLATE_IMPL(_templateParameters, _type) \
    using __thisIsHereToForceASemicolonAfterThisMacro UNUSED_TYPE_ALIAS = int

#else // !USE(SYSTEM_MALLOC) && USE(ISO_MALLOC) && !USE(TZONE_MALLOC)

#include <bmalloc/IsoHeap.h>

// class allocators with IsoHeap fallback if TZoneHeap is disabled, but IsoHeap is enabled.
#define WTF_MAKE_TZONE_OR_ISO_ALLOCATED(name) MAKE_BISO_MALLOCED(name, IsoHeap, WTF_NOEXPORT)
#define WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(name, exportMacro) MAKE_BISO_MALLOCED(name, IsoHeap, exportMacro)

// template allocators with IsoHeap fallback if TZoneHeap is disabled, but IsoHeap is enabled.
#define WTF_MAKE_TZONE_OR_ISO_ALLOCATED_TEMPLATE(name) MAKE_BISO_MALLOCED(name, IsoHeap, WTF_NOEXPORT)

// special class (e.g. those used with CompactPtr) allocators with IsoHeap fallback if TZoneHeap is disabled, but IsoHeap is enabled.
#define WTF_MAKE_COMPACT_TZONE_OR_ISO_ALLOCATED(name) \
    WTF_ALLOW_COMPACT_POINTERS; \
    MAKE_BISO_MALLOCED(name, CompactIsoHeap, WTF_NOEXPORT)
#define WTF_MAKE_COMPACT_TZONE_OR_ISO_ALLOCATED_EXPORT(name, exportMacro) \
    WTF_ALLOW_COMPACT_POINTERS; \
    MAKE_BISO_MALLOCED(name, CompactIsoHeap, exportMacro)

// template implementation to go with WTF_MAKE_TZONE_OR_ISO_ALLOCATED_TEMPLATE
// if TZoneHeap is disabled, but IsoHeap is enabled. This should be added immediately
// after the template definition.
#define WTF_MAKE_TZONE_OR_ISO_ALLOCATED_TEMPLATE_IMPL(_templateParameters, _type) \
    MAKE_BISO_MALLOCED_TEMPLATE_IMPL(_templateParameters, _type)

#endif // USE(SYSTEM_MALLOC) || !USE(ISO_MALLOC)

// template implementation to go with WTF_MAKE_TZONE_ALLOCATED_TEMPLATE and
// WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_EXPORT if TZoneHeap is disabled. This
// should be added immediately after the template definition.
#define WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_IMPL(_templateParameters, _type) \
    using __thisIsHereToForceASemicolonAfterThisMacro UNUSED_TYPE_ALIAS = int

#define WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_IMPL_WITH_MULTIPLE_OR_SPECIALIZED_PARAMETERS() \
    using __thisIsHereToForceASemicolonAfterThisMacro UNUSED_TYPE_ALIAS = int

#else // !USE(SYSTEM_MALLOC) && USE(TZONE_MALLOC)

#include <bmalloc/TZoneHeap.h>

#if !BUSE(TZONE)
#error "TZones enabled in WTF, but not enabled in bmalloc"
#endif

// FastMalloc fallback allocators

// class allocators with FastMalloc fallback if TZoneHeap is enabled.
#define WTF_MAKE_TZONE_ALLOCATED(name) MAKE_BTZONE_MALLOCED(name, NonCompact, WTF_NOEXPORT)
#define WTF_MAKE_TZONE_ALLOCATED_EXPORT(name, exportMacro) MAKE_BTZONE_MALLOCED(name, NonCompact, exportMacro)

// struct allocators with FastMalloc fallback if TZoneHeap is enabled.
#define WTF_MAKE_STRUCT_TZONE_ALLOCATED(name) MAKE_STRUCT_BTZONE_MALLOCED(name, NonCompact, WTF_NOEXPORT)

// template allocators with FastMalloc fallback if TZoneHeap is enabled.
#define WTF_MAKE_TZONE_ALLOCATED_TEMPLATE(name) MAKE_BTZONE_MALLOCED_TEMPLATE(name, NonCompact, WTF_NOEXPORT)
#define WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_EXPORT(name, exportMacro) MAKE_BTZONE_MALLOCED_TEMPLATE(name, NonCompact, exportMacro)

// special class (e.g. those used with CompactPtr) allocators with FastMalloc fallback if TZoneHeap is enabled.
#define WTF_MAKE_COMPACT_TZONE_ALLOCATED(name) MAKE_BTZONE_MALLOCED(name, Compact, WTF_NOEXPORT)
#define WTF_MAKE_COMPACT_TZONE_ALLOCATED_EXPORT(name, exportMacro) MAKE_BTZONE_MALLOCED(name, Compact, exportMacro)

// IsoHeap fallback allocators

// class allocators with IsoHeap fallback if TZoneHeap is enabled.
#define WTF_MAKE_TZONE_OR_ISO_ALLOCATED(name) MAKE_BTZONE_MALLOCED(name, NonCompact, WTF_NOEXPORT)
#define WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(name, exportMacro) MAKE_BTZONE_MALLOCED(name, NonCompact, exportMacro)

// template allocators with IsoHeap fallback if TZoneHeap is enabled.
#define WTF_MAKE_TZONE_OR_ISO_ALLOCATED_TEMPLATE(name) MAKE_BTZONE_MALLOCED_TEMPLATE(name, NonCompact, WTF_NOEXPORT)

// special class (e.g. those used with CompactPtr) allocators with IsoHeap fallback if TZoneHeap is enabled.
#define WTF_MAKE_COMPACT_TZONE_OR_ISO_ALLOCATED(name) \
    WTF_ALLOW_COMPACT_POINTERS; \
    MAKE_BTZONE_MALLOCED(name, Compact, WTF_NOEXPORT)
#define WTF_MAKE_COMPACT_TZONE_OR_ISO_ALLOCATED_EXPORT(name, exportMacro) \
    WTF_ALLOW_COMPACT_POINTERS; \
    MAKE_BTZONE_MALLOCED(name, Compact, exportMacro)

// Template implementations for instantiating allocator template static / methods

// template implementation to go with WTF_MAKE_TZONE_ALLOCATED_TEMPLATE and
// WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_EXPORT if TZoneHeap is enabled. This
// should be added immediately after the template definition.
#define WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_IMPL(_templateParameters, _type) MAKE_BTZONE_MALLOCED_TEMPLATE_IMPL(_templateParameters, _type)

// template implementation to go with WTF_MAKE_TZONE_OR_ISO_ALLOCATED_TEMPLATE
// if TZoneHeap is enabled. This should be added immediately after the template definition.
#define WTF_MAKE_TZONE_OR_ISO_ALLOCATED_TEMPLATE_IMPL(_templateParameters, _type) MAKE_BTZONE_MALLOCED_TEMPLATE_IMPL(_templateParameters, _type)

// template implementation for to go with WTF_MAKE_TZONE_ALLOCATED_TEMPLATE and
// WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_EXPORT if TZoneHeap is ensabled. This
// should be added immediately after the template definition. This version is
// needed in order to support templates with multiple parameters (which
// WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_IMPL cannot support).
//
// Requires the client to define these 3 macros:
//     TZONE_TEMPLATE_PARAMS, TZONE_TYPE
#define WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_IMPL_WITH_MULTIPLE_OR_SPECIALIZED_PARAMETERS() \
    MAKE_BTZONE_MALLOCED_TEMPLATE_IMPL_WITH_MULTIPLE_PARAMETERS()

#endif

// Annotation to forbid use with dynamic allocation

// class / struct which should not use dynamic allocation.
#define WTF_MAKE_TZONE_NON_HEAP_ALLOCATABLE(name) WTF_FORBID_HEAP_ALLOCATION

// class / struct which should not use dynamic allocation. These used to be ISO_ALLOCATED.
// FIXME: we should remove this and use WTF_MAKE_TZONE_NON_HEAP_ALLOCATABLE instead.
#define WTF_MAKE_TZONE_OR_ISO_NON_HEAP_ALLOCATABLE(name) WTF_FORBID_HEAP_ALLOCATION
