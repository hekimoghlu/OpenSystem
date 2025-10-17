/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 30, 2023.
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
#include "util.h"

#include "libyasm-stdint.h"
#include "coretype.h"

#include "errwarn.h"
#include "intnum.h"
#include "expr.h"
#include "value.h"

#include "bytecode.h"


typedef struct bytecode_reserve {
    /*@only@*/ /*@null@*/ yasm_expr *numitems; /* number of items to reserve */
    unsigned int itemsize;          /* size of each item (in bytes) */
} bytecode_reserve;

static void bc_reserve_destroy(void *contents);
static void bc_reserve_print(const void *contents, FILE *f, int indent_level);
static void bc_reserve_finalize(yasm_bytecode *bc, yasm_bytecode *prev_bc);
static int bc_reserve_elem_size(yasm_bytecode *bc);
static int bc_reserve_calc_len(yasm_bytecode *bc,
                               yasm_bc_add_span_func add_span,
                               void *add_span_data);
static int bc_reserve_tobytes(yasm_bytecode *bc, unsigned char **bufp,
                              unsigned char *bufstart, void *d,
                              yasm_output_value_func output_value,
                              /*@null@*/ yasm_output_reloc_func output_reloc);

static const yasm_bytecode_callback bc_reserve_callback = {
    bc_reserve_destroy,
    bc_reserve_print,
    bc_reserve_finalize,
    bc_reserve_elem_size,
    bc_reserve_calc_len,
    yasm_bc_expand_common,
    bc_reserve_tobytes,
    YASM_BC_SPECIAL_RESERVE
};


static void
bc_reserve_destroy(void *contents)
{
    bytecode_reserve *reserve = (bytecode_reserve *)contents;
    yasm_expr_destroy(reserve->numitems);
    yasm_xfree(contents);
}

static void
bc_reserve_print(const void *contents, FILE *f, int indent_level)
{
    const bytecode_reserve *reserve = (const bytecode_reserve *)contents;
    fprintf(f, "%*s_Reserve_\n", indent_level, "");
    fprintf(f, "%*sNum Items=", indent_level, "");
    yasm_expr_print(reserve->numitems, f);
    fprintf(f, "\n%*sItem Size=%u\n", indent_level, "", reserve->itemsize);
}

static void
bc_reserve_finalize(yasm_bytecode *bc, yasm_bytecode *prev_bc)
{
    bytecode_reserve *reserve = (bytecode_reserve *)bc->contents;
    /* multiply reserve expression into multiple */
    if (!bc->multiple)
        bc->multiple = reserve->numitems;
    else
        bc->multiple = yasm_expr_create_tree(bc->multiple, YASM_EXPR_MUL,
                                             reserve->numitems, bc->line);
    reserve->numitems = NULL;
}

static int
bc_reserve_elem_size(yasm_bytecode *bc)
{
    bytecode_reserve *reserve = (bytecode_reserve *)bc->contents;
    return reserve->itemsize;
}

static int
bc_reserve_calc_len(yasm_bytecode *bc, yasm_bc_add_span_func add_span,
                    void *add_span_data)
{
    bytecode_reserve *reserve = (bytecode_reserve *)bc->contents;
    bc->len += reserve->itemsize;
    return 0;
}

static int
bc_reserve_tobytes(yasm_bytecode *bc, unsigned char **bufp,
                   unsigned char *bufstart, void *d,
                   yasm_output_value_func output_value,
                   /*@unused@*/ yasm_output_reloc_func output_reloc)
{
    yasm_internal_error(N_("bc_reserve_tobytes called"));
    /*@notreached@*/
    return 1;
}

yasm_bytecode *
yasm_bc_create_reserve(yasm_expr *numitems, unsigned int itemsize,
                       unsigned long line)
{
    bytecode_reserve *reserve = yasm_xmalloc(sizeof(bytecode_reserve));

    /*@-mustfree@*/
    reserve->numitems = numitems;
    /*@=mustfree@*/
    reserve->itemsize = itemsize;

    return yasm_bc_create_common(&bc_reserve_callback, reserve, line);
}

const yasm_expr *
yasm_bc_reserve_numitems(yasm_bytecode *bc, unsigned int *itemsize)
{
    bytecode_reserve *reserve;

    if (bc->callback != &bc_reserve_callback)
        return NULL;

    reserve = (bytecode_reserve *)bc->contents;
    *itemsize = reserve->itemsize;
    return reserve->numitems;
}
