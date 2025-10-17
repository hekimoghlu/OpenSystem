/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 3, 2022.
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

#include "dwarf2-dbgfmt.h"


static void
dwarf2_append_arange(yasm_section *debug_aranges, /*@only@*/ yasm_expr *start,
                     /*@only@*/ yasm_expr *length, unsigned int sizeof_address)
{
    yasm_datavalhead dvs;
    yasm_bytecode *bc;

    yasm_dvs_initialize(&dvs);
    yasm_dvs_append(&dvs, yasm_dv_create_expr(start));
    yasm_dvs_append(&dvs, yasm_dv_create_expr(length));
    bc = yasm_bc_create_data(&dvs, sizeof_address, 0, NULL, 0);
    yasm_bc_finalize(bc, yasm_dwarf2__append_bc(debug_aranges, bc));
    yasm_bc_calc_len(bc, NULL, NULL);
}

typedef struct dwarf2_aranges_info {
    yasm_section *debug_aranges;    /* section to which address ranges go */
    yasm_object *object;
    yasm_dbgfmt_dwarf2 *dbgfmt_dwarf2;
} dwarf2_aranges_info;

static int
dwarf2_generate_aranges_section(yasm_section *sect, /*@null@*/ void *d)
{
    dwarf2_aranges_info *info = (dwarf2_aranges_info *)d;
    yasm_dbgfmt_dwarf2 *dbgfmt_dwarf2 = info->dbgfmt_dwarf2;
    /*@null@*/ dwarf2_section_data *dsd;
    /*@only@*/ yasm_expr *start, *length;

    dsd = yasm_section_get_data(sect, &yasm_dwarf2__section_data_cb);
    if (!dsd)
        return 0;       /* no line data for this section */

    /* Create address range descriptor */
    start = yasm_expr_create_ident(
        yasm_expr_sym(yasm_dwarf2__bc_sym(info->object->symtab,
                                          yasm_section_bcs_first(sect))), 0);
    length = yasm_expr_create_ident(
        yasm_expr_int(yasm_calc_bc_dist(yasm_section_bcs_first(sect),
                                        yasm_section_bcs_last(sect))), 0);
    dwarf2_append_arange(info->debug_aranges, start, length,
                         dbgfmt_dwarf2->sizeof_address);

    return 0;
}

yasm_section *
yasm_dwarf2__generate_aranges(yasm_object *object, yasm_section *debug_info)
{
    yasm_dbgfmt_dwarf2 *dbgfmt_dwarf2 = (yasm_dbgfmt_dwarf2 *)object->dbgfmt;
    int new;
    yasm_section *debug_aranges;
    yasm_bytecode *bc;
    dwarf2_head *head;
    dwarf2_aranges_info info;

    debug_aranges =
        yasm_object_get_general(object, ".debug_aranges",
                                2*dbgfmt_dwarf2->sizeof_address, 0, 0, &new,
                                0);

    /* header */
    head = yasm_dwarf2__add_head(dbgfmt_dwarf2, debug_aranges, debug_info, 1,
                                 1);

    /* align ranges to 2x address size (range size) */
    bc = yasm_bc_create_align(
        yasm_expr_create_ident(yasm_expr_int(
            yasm_intnum_create_uint(dbgfmt_dwarf2->sizeof_address*2)), 0),
        NULL, NULL, NULL, 0);
    yasm_bc_finalize(bc, yasm_dwarf2__append_bc(debug_aranges, bc));
    yasm_bc_calc_len(bc, NULL, NULL);

    info.debug_aranges = debug_aranges;
    info.object = object;
    info.dbgfmt_dwarf2 = dbgfmt_dwarf2;

    yasm_object_sections_traverse(object, (void *)&info,
                                  dwarf2_generate_aranges_section);

    /* Terminate with empty address range descriptor */
    dwarf2_append_arange(debug_aranges,
        yasm_expr_create_ident(yasm_expr_int(yasm_intnum_create_uint(0)), 0),
        yasm_expr_create_ident(yasm_expr_int(yasm_intnum_create_uint(0)), 0),
        dbgfmt_dwarf2->sizeof_address);

    /* mark end of aranges information */
    yasm_dwarf2__set_head_end(head, yasm_section_bcs_last(debug_aranges));

    return debug_aranges;
}

