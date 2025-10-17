/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 24, 2025.
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

#include <wtf/FastMalloc.h>
#include <wtf/ForbidHeapAllocation.h>
#include <wtf/Platform.h>

#if USE(SYSTEM_MALLOC) || !USE(ISO_MALLOC)

#define WTF_MAKE_ISO_ALLOCATED(name) WTF_MAKE_FAST_ALLOCATED
#define WTF_MAKE_ISO_ALLOCATED_EXPORT(name, exportMacro) WTF_MAKE_FAST_ALLOCATED
#define WTF_MAKE_COMPACT_ISO_ALLOCATED(name) WTF_MAKE_FAST_COMPACT_ALLOCATED
#define WTF_MAKE_COMPACT_ISO_ALLOCATED_EXPORT(name, exportMacro) WTF_MAKE_FAST_COMPACT_ALLOCATED
#define WTF_MAKE_ISO_NONALLOCATABLE(name) WTF_FORBID_HEAP_ALLOCATION

#else

#include <bmalloc/IsoHeap.h>

#define WTF_NOEXPORT

#define WTF_MAKE_ISO_ALLOCATED(name) MAKE_BISO_MALLOCED(name, IsoHeap, WTF_NOEXPORT)
#define WTF_MAKE_ISO_ALLOCATED_EXPORT(name, exportMacro) MAKE_BISO_MALLOCED(name, IsoHeap, exportMacro)
#define WTF_MAKE_COMPACT_ISO_ALLOCATED(name) \
    WTF_ALLOW_COMPACT_POINTERS; \
    MAKE_BISO_MALLOCED(name, CompactIsoHeap, WTF_NOEXPORT)
#define WTF_MAKE_COMPACT_ISO_ALLOCATED_EXPORT(name, exportMacro) \
    WTF_ALLOW_COMPACT_POINTERS; \
    MAKE_BISO_MALLOCED(name, CompactIsoHeap, exportMacro)
#define WTF_MAKE_ISO_NONALLOCATABLE(name) WTF_FORBID_HEAP_ALLOCATION

#endif

