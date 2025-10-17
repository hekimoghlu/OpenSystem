/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 19, 2025.
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
#ifndef PAS_MMAP_CAPABILITY_H
#define PAS_MMAP_CAPABILITY_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

/* Libpas sometimes allocates its own memory from the OS, and sometimes is told to manage memory already
   allocated by someone else. Sometimes, clients don't want libpas to use certaint memory management functions
   on the memory that they ask libpas to manage. This enumeration controls what memory management functions
   libpas can call for some memory that it manages. */
enum pas_mmap_capability {
    /* Indicates that libpas isn't allowed to call mmap, mprotect, or most other memory management function on
       the given memory. This is used for memory that was allocated by the client with some specific permissions
       or other special mmap arguments. Or, maybe it wasn't even allocated with mmap.

       Libpas may still call madvise() on memory with this capability, since madvise() doesn't change the
       permissions of the mapping. Currently, madvise() is the only syscall that libpas is allowed to use for
       memory that is pas_may_not_mmap.
    
       Using pas_may_not_mmap may degrade peformance, affect the accuracy of OS memory usage accounting, or turn
       off some runtime checking. But it will not affect the correctness of the algorithm.
    
       Currently the primary client of this option is jit_heap_config, since the JIT heap is mmapped in a
       special way and with special permissions. */
    pas_may_not_mmap,

    /* Indicates that libpas should assume that the memory is mmapped in some canonical way (anonymous, RW but
       not X, etc). Libpas may call mmap, mprotect and other such functions on this memory so long as it
       returns it to the canonical state before the memory is returned by some allocate function. Libpas could
       use this capability to:
       
       - Rapidly zero-fill the memory with a fixed mmap call.
       - Reset the commit state with a fixed mmap call.
       - Perform runtime checking that decommitted memory is not used by mprotecting it to PROT_NONE (and then
         returning it to normal permissions when recommitted).
    
       The capabilities enabled by pas_may_mmap are good for performance and testing, but aren't necessary for
       libpas to work correctly, so it's not the end of the world to have to use pas_may_not_mmap instead. */
    pas_may_mmap
};

typedef enum pas_mmap_capability pas_mmap_capability;

static inline const char* pas_mmap_capability_get_string(pas_mmap_capability capability)
{
    switch (capability) {
    case pas_may_not_mmap:
        return "may_not_mmap";
    case pas_may_mmap:
        return "may_mmap";
    }
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

PAS_END_EXTERN_C;

#endif /* PAS_MMAP_CAPABILITY_H */

