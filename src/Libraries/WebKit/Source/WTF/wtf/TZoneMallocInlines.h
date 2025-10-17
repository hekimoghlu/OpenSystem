/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 21, 2022.
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

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

#if USE(SYSTEM_MALLOC) || !USE(TZONE_MALLOC)

#include <wtf/FastMalloc.h>

#define WTF_MAKE_TZONE_ALLOCATED_INLINE(typeName) WTF_MAKE_FAST_ALLOCATED

#define WTF_MAKE_TZONE_ALLOCATED_IMPL(typeName) using __makeTZoneMallocedMacroSemicolonifier UNUSED_TYPE_ALIAS = int

#define WTF_MAKE_COMPACT_TZONE_ALLOCATED_IMPL(typeName) using __makeTZoneMallocedMacroSemicolonifier UNUSED_TYPE_ALIAS = int

#define WTF_MAKE_STRUCT_TZONE_ALLOCATED_IMPL(typeName) WTF_MAKE_TZONE_ALLOCATED_IMPL(typeName)

#if USE(SYSTEM_MALLOC) || !USE(ISO_MALLOC)

#define WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(name) using __makeTZoneMallocedMacroSemicolonifier UNUSED_TYPE_ALIAS = int
#define WTF_MAKE_COMPACT_TZONE_OR_ISO_ALLOCATED_IMPL(name) using __makeTZoneMallocedMacroSemicolonifier UNUSED_TYPE_ALIAS = int

#else // !USE(SYSTEM_MALLOC) && USE(ISO_MALLOC) && !USE(TZONE_MALLOC)

#include <bmalloc/IsoHeapInlines.h>

#define WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(name) MAKE_BISO_MALLOCED_IMPL(name, IsoHeap)
#define WTF_MAKE_COMPACT_TZONE_OR_ISO_ALLOCATED_IMPL(name) MAKE_BISO_MALLOCED_IMPL(name, CompactIsoHeap)

#endif

#else // !USE(SYSTEM_MALLOC) && USE(TZONE_MALLOC)

#include <bmalloc/TZoneHeapInlines.h>

#if !BUSE(TZONE)
#error "TZones enabled in WTF, but not enabled in bmalloc"
#endif

#define WTF_MAKE_TZONE_ALLOCATED_INLINE(type) MAKE_BTZONE_MALLOCED_INLINE(type, NonCompact)

#define WTF_MAKE_TZONE_ALLOCATED_IMPL(type) MAKE_BTZONE_MALLOCED_IMPL(type, NonCompact)

#define WTF_MAKE_COMPACT_TZONE_ALLOCATED_IMPL(type) MAKE_BTZONE_MALLOCED_IMPL(type, Compact)

#define WTF_MAKE_STRUCT_TZONE_ALLOCATED_IMPL(typeName) \
    MAKE_BTZONE_MALLOCED_IMPL(typeName, NonCompact)

#define WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(name) MAKE_BTZONE_MALLOCED_IMPL(name, NonCompact)
#define WTF_MAKE_COMPACT_TZONE_OR_ISO_ALLOCATED_IMPL(name) MAKE_BTZONE_MALLOCED_IMPL(name, Compact)

#endif

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
