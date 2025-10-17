/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 21, 2024.
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

#include "Block_private.h"

#if HAVE_UNWIND

typedef struct Block_layout Block_layout;

// These functions were split out from runtime.cpp to avoid having to link to
// libc++ as doing so would create a circular dependencies between dylibs.
// Compiling this file with -fexceptions requires linking to libunwind and
// libcompiler_rt, both of which are in libSystem.


void _call_copy_helpers_excp(Block_layout *dstbl, Block_layout *srcbl,
                             HelperBaseData *helper) {
    ExcpCleanupInfo __attribute__((cleanup(_cleanup_generic_captures)))
    info = {EXCP_NONE, dstbl, helper};
    // helper is null if generic helpers aren't used.
    if (helper) {
        info.state = EXCP_COPY_GENERIC;
        _call_generic_copy_helper(dstbl, srcbl, helper);
    }
    info.state = EXCP_COPY_CUSTOM;
    _call_custom_copy_helper(dstbl, srcbl);
    info.state = EXCP_NONE;
}

void _call_dispose_helpers_excp(Block_layout *bl, HelperBaseData *helper) {
    ExcpCleanupInfo __attribute__((cleanup(_cleanup_generic_captures)))
    info = {EXCP_DESTROY_CUSTOM, bl, helper};
    _call_custom_dispose_helper(bl);
    // helper is null if generic helpers aren't used.
    if (helper) {
        info.state = EXCP_DESTROY_GENERIC;
        _call_generic_destroy_helper(bl, helper);
    }
    info.state = EXCP_NONE;
}

#endif
