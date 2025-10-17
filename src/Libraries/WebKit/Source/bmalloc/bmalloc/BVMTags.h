/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 14, 2024.
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

#include "BPlatform.h"

#if BPLATFORM(PLAYSTATION)
#include <sys/mman.h>
#endif

// On Mac OS X, the VM subsystem allows tagging memory requested from mmap and vm_map
// in order to aid tools that inspect system memory use.
#if BOS(DARWIN)

#include <mach/vm_statistics.h>

#if defined(VM_MEMORY_TCMALLOC)
#define VM_TAG_FOR_TCMALLOC_MEMORY VM_MAKE_TAG(VM_MEMORY_TCMALLOC)
#else
#define VM_TAG_FOR_TCMALLOC_MEMORY VM_MAKE_TAG(53)
#endif // defined(VM_MEMORY_TCMALLOC)

#if defined(VM_MEMORY_JAVASCRIPT_JIT_EXECUTABLE_ALLOCATOR)
#define VM_TAG_FOR_EXECUTABLEALLOCATOR_MEMORY VM_MAKE_TAG(VM_MEMORY_JAVASCRIPT_JIT_EXECUTABLE_ALLOCATOR)
#else
#define VM_TAG_FOR_EXECUTABLEALLOCATOR_MEMORY VM_MAKE_TAG(64)
#endif // defined(VM_MEMORY_JAVASCRIPT_JIT_EXECUTABLE_ALLOCATOR)

#if defined(VM_MEMORY_JAVASCRIPT_JIT_REGISTER_FILE)
#define VM_TAG_FOR_ISOHEAP_MEMORY VM_MAKE_TAG(VM_MEMORY_JAVASCRIPT_JIT_REGISTER_FILE)
#else
#define VM_TAG_FOR_ISOHEAP_MEMORY VM_MAKE_TAG(65)
#endif // defined(VM_MEMORY_JAVASCRIPT_JIT_REGISTER_FILE)

#if defined(VM_MEMORY_JAVASCRIPT_CORE)
#define VM_TAG_FOR_GIGACAGE_MEMORY VM_MAKE_TAG(VM_MEMORY_JAVASCRIPT_CORE)
#else
#define VM_TAG_FOR_GIGACAGE_MEMORY VM_MAKE_TAG(63)
#endif // defined(VM_MEMORY_JAVASCRIPT_CORE)

#elif BPLATFORM(PLAYSTATION) && defined(VM_MAKE_TAG)

#define VM_TAG_FOR_TCMALLOC_MEMORY VM_MAKE_TAG(VM_TYPE_USER1)
#define VM_TAG_FOR_ISOHEAP_MEMORY VM_MAKE_TAG(VM_TYPE_USER2)
#define VM_TAG_FOR_EXECUTABLEALLOCATOR_MEMORY VM_MAKE_TAG(VM_TYPE_USER3)
#define VM_TAG_FOR_GIGACAGE_MEMORY VM_MAKE_TAG(VM_TYPE_USER4)

#else // BOS(DARWIN)

#define VM_TAG_FOR_TCMALLOC_MEMORY -1
#define VM_TAG_FOR_GIGACAGE_MEMORY -1
#define VM_TAG_FOR_EXECUTABLEALLOCATOR_MEMORY -1
#define VM_TAG_FOR_ISOHEAP_MEMORY -1

#endif // BOS(DARWIN)

namespace bmalloc {

enum class VMTag {
    Unknown = -1,
    Malloc = VM_TAG_FOR_TCMALLOC_MEMORY,
    IsoHeap = VM_TAG_FOR_ISOHEAP_MEMORY,
    JSJITCode = VM_TAG_FOR_EXECUTABLEALLOCATOR_MEMORY,
    JSGigacage = VM_TAG_FOR_GIGACAGE_MEMORY,
};

} // namespace bmalloc
