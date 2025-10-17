/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 2, 2025.
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

#include "BExport.h"
#include <atomic>

#define BFOR_EACH_ALLOCATION_KIND(macro) \
    macro(JS_CELL)      /* Allocation of any JSCell */ \
    macro(NON_JS_CELL)  /* Allocation of any non-JSCell */ \
    macro(GIGACAGE)     /* Allocation within a gigacage, takes heap kind as first parameter */ \
    macro(COMPACTIBLE)  /* Pointers to this allocation are allowed to be stored compact. */

#define BPROFILE_ALLOCATION(kind, ...) \
    BPROFILE_ALLOCATION_ ## kind(__VA_ARGS__)

#define BPROFILE_TRY_ALLOCATION(kind, ...) \
    BPROFILE_TRY_ALLOCATION_ ## kind(__VA_ARGS__)

// Definitions of specializations of the above macro (i.e. BPROFILE_ALLOCATION_JS_CELL(ptr, size))
// may be provided in an AllocationCountsAdditions.h header to add custom profiling code
// at an allocation site, taking both the pointer variable (which may be modified) and the
// size of the allocation in bytes.

#if __has_include(<WebKitAdditions/AllocationCountsAdditions.h>)
#include <WebKitAdditions/AllocationCountsAdditions.h>
#elif __has_include(<AllocationCountsAdditions.h>)
#include <AllocationCountsAdditions.h>
#endif

// If allocation profiling macros weren't defined above, we define them below as no-ops.
// Additionally, BENABLE(PROFILE_<type>_ALLOCATION) can be queried to see if we are profiling
// allocations of a specific kind.

#ifdef BPROFILE_ALLOCATION_JS_CELL
#define BENABLE_PROFILE_JS_CELL_ALLOCATION 1
#else
#define BENABLE_PROFILE_JS_CELL_ALLOCATION 0
#define BPROFILE_ALLOCATION_JS_CELL(ptr, size) do { } while (false)
#endif

#ifdef BPROFILE_TRY_ALLOCATION_JS_CELL
#define BENABLE_PROFILE_JS_CELL_TRY_ALLOCATION 1
#else
#define BENABLE_PROFILE_JS_CELL_TRY_ALLOCATION 0
#define BPROFILE_TRY_ALLOCATION_JS_CELL(ptr, size) do { } while (false)
#endif

#ifdef BPROFILE_ALLOCATION_NON_JS_CELL
#define BENABLE_PROFILE_NON_JS_CELL_ALLOCATION 1
#else
#define BENABLE_PROFILE_NON_JS_CELL_ALLOCATION 0
#define BPROFILE_ALLOCATION_NON_JS_CELL(ptr, size) do { } while (false)
#endif

#ifdef BPROFILE_TRY_ALLOCATION_NON_JS_CELL
#define BENABLE_PROFILE_NON_JS_CELL_TRY_ALLOCATION 1
#else
#define BENABLE_PROFILE_NON_JS_CELL_TRY_ALLOCATION 0
#define BPROFILE_TRY_ALLOCATION_NON_JS_CELL(ptr, size) do { } while (false)
#endif

#ifdef BPROFILE_ALLOCATION_GIGACAGE
#define BENABLE_PROFILE_GIGACAGE_ALLOCATION 1
#else
#define BENABLE_PROFILE_GIGACAGE_ALLOCATION 0
#define BPROFILE_ALLOCATION_GIGACAGE(kind, ptr, size) do { } while (false)
#endif

#ifdef BPROFILE_TRY_ALLOCATION_GIGACAGE
#define BENABLE_PROFILE_GIGACAGE_TRY_ALLOCATION 1
#else
#define BENABLE_PROFILE_GIGACAGE_TRY_ALLOCATION 0
#define BPROFILE_TRY_ALLOCATION_GIGACAGE(kind, ptr, size) do { } while (false)
#endif

#ifdef BPROFILE_ALLOCATION_COMPACTIBLE
#define BENABLE_PROFILE_COMPACTIBLE_ALLOCATION 1
#else
#define BENABLE_PROFILE_COMPACTIBLE_ALLOCATION 0
#define BPROFILE_ALLOCATION_COMPACTIBLE(ptr, size) do { } while (false)
#endif

#ifdef BPROFILE_TRY_ALLOCATION_COMPACTIBLE
#define BENABLE_PROFILE_COMPACTIBLE_TRY_ALLOCATION 1
#else
#define BENABLE_PROFILE_COMPACTIBLE_TRY_ALLOCATION 0
#define BPROFILE_TRY_ALLOCATION_COMPACTIBLE(ptr, size) do { } while (false)
#endif

#ifdef BPROFILE_ZERO_FILL_PAGE
#define BENABLE_PROFILE_ZERO_FILL_PAGE 1
#else
#define BENABLE_PROFILE_ZERO_FILL_PAGE 0
#define BPROFILE_ZERO_FILL_PAGE(ptr, size, flags, tag) do { } while (false)
#endif

#ifdef BPROFILE_ALLOCATION_VM_ALLOCATION
#define BENABLE_PROFILE_VM_ALLOCATION 1
#else
#define BENABLE_PROFILE_VM_ALLOCATION 0
#define BPROFILE_ALLOCATION_VM_ALLOCATION(size, usage) do { } while (false)
#endif

#ifdef BPROFILE_ALLOCATION_INITIAL_GIGACAGE
#define BENABLE_PROFILE_INITIAL_GIGACAGE_ALLOCATION 1
#else
#define BENABLE_PROFILE_INITIAL_GIGACAGE_ALLOCATION 0
#define BPROFILE_ALLOCATION_INITIAL_GIGACAGE(size) do { } while (false)
#endif
