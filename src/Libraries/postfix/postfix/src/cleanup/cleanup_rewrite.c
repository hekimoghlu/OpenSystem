/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 7, 2023.
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

#include <msg.h>
#include <vstring.h>

/* Global library. */

#include <tok822.h>
#include <rewrite_clnt.h>
#include <quote_822_local.h>

/* Application-specific. */

#include "cleanup.h"

#define STR		vstring_str

/* cleanup_rewrite_external - rewrite address external form */

int     cleanup_rewrite_external(const char *context_name, VSTRING *result,
				         const char *addr)
{
    rewrite_clnt(context_name, addr, result);
    return (strcmp(STR(result), addr) != 0);
}

/* cleanup_rewrite_tree - rewrite address node */

int    cleanup_rewrite_tree(const char *context_name, TOK822 *tree)
{
    VSTRING *dst = vstring_alloc(100);
    VSTRING *src = vstring_alloc(100);
    int     did_rewrite;

    tok822_externalize(src, tree->head, TOK822_STR_DEFL);
    did_rewrite = cleanup_rewrite_external(context_name, dst, STR(src));
    tok822_free_tree(tree->head);
    tree->head = tok822_scan(STR(dst), &tree->tail);
    vstring_free(dst);
    vstring_free(src);
    return (did_rewrite);
}

/* cleanup_rewrite_internal - rewrite address internal form */

int     cleanup_rewrite_internal(const char *context_name,
				         VSTRING *result, const char *addr)
{
    VSTRING *dst = vstring_alloc(100);
    VSTRING *src = vstring_alloc(100);
    int     did_rewrite;

    quote_822_local(src, addr);
    did_rewrite = cleanup_rewrite_external(context_name, dst, STR(src));
    unquote_822_local(result, STR(dst));
    vstring_free(dst);
    vstring_free(src);
    return (did_rewrite);
}
