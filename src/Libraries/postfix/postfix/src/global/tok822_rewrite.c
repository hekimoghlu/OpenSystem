/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 30, 2023.
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
#include <msg.h>

/* Global library. */

#include "rewrite_clnt.h"
#include "tok822.h"

/* tok822_rewrite - address rewriting interface */

TOK822 *tok822_rewrite(TOK822 *addr, const char *how)
{
    VSTRING *input_ext_form = vstring_alloc(100);
    VSTRING *canon_ext_form = vstring_alloc(100);

    if (addr->type != TOK822_ADDR)
	msg_panic("tok822_rewrite: non-address token type: %d", addr->type);

    /*
     * Externalize the token tree, ship it to the rewrite service, and parse
     * the result. Shipping external form is much simpler than shipping parse
     * trees.
     */
    tok822_externalize(input_ext_form, addr->head, TOK822_STR_DEFL);
    if (msg_verbose)
	msg_info("tok822_rewrite: input: %s", vstring_str(input_ext_form));
    rewrite_clnt(how, vstring_str(input_ext_form), canon_ext_form);
    if (msg_verbose)
	msg_info("tok822_rewrite: result: %s", vstring_str(canon_ext_form));
    tok822_free_tree(addr->head);
    addr->head = tok822_scan(vstring_str(canon_ext_form), &addr->tail);

    vstring_free(input_ext_form);
    vstring_free(canon_ext_form);
    return (addr);
}
