/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 11, 2021.
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
#ifndef COFF_OBJFMT_H
#define COFF_OBJFMT_H

typedef struct coff_unwind_code {
    SLIST_ENTRY(coff_unwind_code) link;

    /*@dependent@*/ yasm_symrec *proc;      /* Start of procedure */
    /*@dependent@*/ yasm_symrec *loc;       /* Location of operation */
    /* Unwind operation code */
    enum {
        UWOP_PUSH_NONVOL = 0,
        UWOP_ALLOC_LARGE = 1,
        UWOP_ALLOC_SMALL = 2,
        UWOP_SET_FPREG = 3,
        UWOP_SAVE_NONVOL = 4,
        UWOP_SAVE_NONVOL_FAR = 5,
        UWOP_SAVE_XMM128 = 8,
        UWOP_SAVE_XMM128_FAR = 9,
        UWOP_PUSH_MACHFRAME = 10
    } opcode;
    unsigned int info;          /* Operation info */
    yasm_value off;             /* Offset expression (used for some codes) */
} coff_unwind_code;

typedef struct coff_unwind_info {
    /*@dependent@*/ yasm_symrec *proc;      /* Start of procedure */
    /*@dependent@*/ yasm_symrec *prolog;    /* End of prologue */

    /*@null@*/ /*@dependent@*/ yasm_symrec *ehandler;   /* Error handler */

    unsigned long framereg;     /* Frame register */
    yasm_value frameoff;        /* Frame offset */

    /* Linked list of codes, in decreasing location offset order.
     * Inserting at the head of this list during assembly naturally results
     * in this sorting.
     */
    SLIST_HEAD(coff_unwind_code_head, coff_unwind_code) codes;

    /* These aren't used until inside of generate. */
    yasm_value prolog_size;
    yasm_value codes_count;
} coff_unwind_info;

coff_unwind_info *yasm_win64__uwinfo_create(void);
void yasm_win64__uwinfo_destroy(coff_unwind_info *info);
void yasm_win64__unwind_generate(yasm_section *xdata,
                                 /*@only@*/ coff_unwind_info *info,
                                 unsigned long line);

#endif
