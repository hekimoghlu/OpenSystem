/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 18, 2022.
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
#include <util.h>

#include <libyasm.h>


yasm_dbgfmt_module yasm_null_LTX_dbgfmt;


static /*@null@*/ /*@only@*/ yasm_dbgfmt *
null_dbgfmt_create(yasm_object *object)
{
    yasm_dbgfmt_base *dbgfmt = yasm_xmalloc(sizeof(yasm_dbgfmt_base));
    dbgfmt->module = &yasm_null_LTX_dbgfmt;
    return (yasm_dbgfmt *)dbgfmt;
}

static void
null_dbgfmt_destroy(/*@only@*/ yasm_dbgfmt *dbgfmt)
{
    yasm_xfree(dbgfmt);
}

static void
null_dbgfmt_generate(yasm_object *object, yasm_linemap *linemap,
                     yasm_errwarns *errwarns)
{
}


/* Define dbgfmt structure -- see dbgfmt.h for details */
yasm_dbgfmt_module yasm_null_LTX_dbgfmt = {
    "No debugging info",
    "null",
    NULL,       /* no directives */
    null_dbgfmt_create,
    null_dbgfmt_destroy,
    null_dbgfmt_generate
};
