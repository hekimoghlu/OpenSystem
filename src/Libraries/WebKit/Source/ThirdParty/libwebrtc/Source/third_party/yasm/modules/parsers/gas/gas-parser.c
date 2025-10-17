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
#include <util.h>

#include <libyasm.h>

#include "gas-parser.h"


static void
gas_parser_do_parse(yasm_object *object, yasm_preproc *pp,
                    int save_input, yasm_linemap *linemap,
                    yasm_errwarns *errwarns)
{
    yasm_parser_gas parser_gas;
    int i;

    parser_gas.object = object;
    parser_gas.linemap = linemap;

    parser_gas.locallabel_base = (char *)NULL;
    parser_gas.locallabel_base_len = 0;

    parser_gas.dir_fileline = 0;
    parser_gas.dir_file = NULL;
    parser_gas.dir_line = 0;
    parser_gas.seen_line_marker = 0;

    parser_gas.preproc = pp;
    parser_gas.errwarns = errwarns;

    parser_gas.prev_bc = yasm_section_bcs_first(object->cur_section);

    parser_gas.save_input = save_input;
    parser_gas.save_last = 0;

    parser_gas.peek_token = NONE;

    parser_gas.line = NULL;

    /* initialize scanner structure */
    yasm_scanner_initialize(&parser_gas.s);

    parser_gas.state = INITIAL;

    for (i=0; i<10; i++)
        parser_gas.local[i] = 0;

    parser_gas.intel_syntax = 0;

    parser_gas.is_cpp_preproc =
        yasm__strcasecmp(((yasm_preproc_base*)pp)->module->keyword, "cpp") == 0;
    parser_gas.is_nasm_preproc =
        yasm__strcasecmp(((yasm_preproc_base*)pp)->module->keyword, "nasm") == 0;

    gas_parser_parse(&parser_gas);

    /* Check for ending inside a comment */
    if (parser_gas.state == COMMENT) {
        yasm_warn_set(YASM_WARN_GENERAL, N_("end of file in comment"));
        /* XXX: Minus two to compensate for already having moved past the EOF
         * in the linemap.
         */
        yasm_errwarn_propagate(errwarns,
                               yasm_linemap_get_current(parser_gas.linemap)-2);
    }

    yasm_scanner_delete(&parser_gas.s);

    /* Free locallabel base if necessary */
    if (parser_gas.locallabel_base)
        yasm_xfree(parser_gas.locallabel_base);

    if (parser_gas.dir_file)
        yasm_xfree(parser_gas.dir_file);

    /* Convert all undefined symbols into extern symbols */
    yasm_symtab_parser_finalize(object->symtab, 1, errwarns);
}

/* Define valid preprocessors to use with this parser */
static const char *gas_parser_preproc_keywords[] = {
    "gas",
    "raw",
    "cpp",
    "nasm",
    NULL
};

/* Define parser structure -- see parser.h for details */
yasm_parser_module yasm_gas_LTX_parser = {
    "GNU AS (GAS)-compatible parser",
    "gas",
    gas_parser_preproc_keywords,
    "gas",
    NULL,   /* No standard macros */
    gas_parser_do_parse
};
yasm_parser_module yasm_gnu_LTX_parser = {
    "GNU AS (GAS)-compatible parser",
    "gnu",
    gas_parser_preproc_keywords,
    "gas",
    NULL,   /* No standard macros */
    gas_parser_do_parse
};
