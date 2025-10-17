/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 23, 2023.
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

#include "cv-dbgfmt.h"

yasm_dbgfmt_module yasm_cv8_LTX_dbgfmt;


static /*@null@*/ /*@only@*/ yasm_dbgfmt *
cv_dbgfmt_create(yasm_object *object, yasm_dbgfmt_module *module, int version)
{
    yasm_dbgfmt_cv *dbgfmt_cv = yasm_xmalloc(sizeof(yasm_dbgfmt_cv));
    size_t i;

    dbgfmt_cv->dbgfmt.module = module;

    dbgfmt_cv->filenames_allocated = 32;
    dbgfmt_cv->filenames_size = 0;
    dbgfmt_cv->filenames =
        yasm_xmalloc(sizeof(cv_filename)*dbgfmt_cv->filenames_allocated);
    for (i=0; i<dbgfmt_cv->filenames_allocated; i++) {
        dbgfmt_cv->filenames[i].pathname = NULL;
        dbgfmt_cv->filenames[i].filename = NULL;
        dbgfmt_cv->filenames[i].str_off = 0;
        dbgfmt_cv->filenames[i].info_off = 0;
    }

    dbgfmt_cv->version = version;

    return (yasm_dbgfmt *)dbgfmt_cv;
}

static /*@null@*/ /*@only@*/ yasm_dbgfmt *
cv8_dbgfmt_create(yasm_object *object)
{
    return cv_dbgfmt_create(object, &yasm_cv8_LTX_dbgfmt, 8);
}

static void
cv_dbgfmt_destroy(/*@only@*/ yasm_dbgfmt *dbgfmt)
{
    yasm_dbgfmt_cv *dbgfmt_cv = (yasm_dbgfmt_cv *)dbgfmt;
    size_t i;
    for (i=0; i<dbgfmt_cv->filenames_size; i++) {
        if (dbgfmt_cv->filenames[i].pathname)
            yasm_xfree(dbgfmt_cv->filenames[i].pathname);
    }
    yasm_xfree(dbgfmt_cv->filenames);
    yasm_xfree(dbgfmt);
}

/* Add a bytecode to a section, updating offset on insertion;
 * no optimization necessary.
 */
yasm_bytecode *
yasm_cv__append_bc(yasm_section *sect, yasm_bytecode *bc)
{
    yasm_bytecode *precbc = yasm_section_bcs_last(sect);
    bc->offset = yasm_bc_next_offset(precbc);
    yasm_section_bcs_append(sect, bc);
    return precbc;
}

static void
cv_dbgfmt_generate(yasm_object *object, yasm_linemap *linemap,
                   yasm_errwarns *errwarns)
{
    yasm_cv__generate_symline(object, linemap, errwarns);
    yasm_cv__generate_type(object);
}

/* Define dbgfmt structure -- see dbgfmt.h for details */
yasm_dbgfmt_module yasm_cv8_LTX_dbgfmt = {
    "CodeView debugging format for VC8",
    "cv8",
    NULL,   /* no directives */
    cv8_dbgfmt_create,
    cv_dbgfmt_destroy,
    cv_dbgfmt_generate
};
