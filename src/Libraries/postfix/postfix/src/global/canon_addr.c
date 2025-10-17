/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 1, 2025.
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
/* System library. */

#include <sys_defs.h>

/* Utility library. */

#include <vstring.h>
#include <mymalloc.h>

/* Global library. */

#include "rewrite_clnt.h"
#include "canon_addr.h"

/* canon_addr_external - make address fully qualified, external form */

VSTRING *canon_addr_external(VSTRING *result, const char *addr)
{
    return (rewrite_clnt(REWRITE_CANON, addr, result));
}

/* canon_addr_internal - make address fully qualified, internal form */

VSTRING *canon_addr_internal(VSTRING *result, const char *addr)
{
    return (rewrite_clnt_internal(REWRITE_CANON, addr, result));
}
