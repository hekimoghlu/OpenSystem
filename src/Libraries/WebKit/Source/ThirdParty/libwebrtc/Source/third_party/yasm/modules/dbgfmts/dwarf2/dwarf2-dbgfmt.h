/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 1, 2025.
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
#ifndef YASM_DWARF2_DBGFMT_H
#define YASM_DWARF2_DBGFMT_H

#define WITH_DWARF3 1

typedef struct {
    char *pathname;         /* full filename */
    char *filename;         /* basename of full filename */
    unsigned long dir;      /* index into directories array for relative path;
                             * 0 for current directory. */
} dwarf2_filename;

/* Global data */
typedef struct yasm_dbgfmt_dwarf2 {
    yasm_dbgfmt_base dbgfmt;        /* base structure */

    char **dirs;
    unsigned long dirs_size;
    unsigned long dirs_allocated;

    dwarf2_filename *filenames;
    unsigned long filenames_size;
    unsigned long filenames_allocated;

    enum {
        DWARF2_FORMAT_32BIT,
        DWARF2_FORMAT_64BIT
    } format;

    unsigned int sizeof_address, sizeof_offset, min_insn_len;
} yasm_dbgfmt_dwarf2;

/* .loc directive data */
typedef struct dwarf2_loc {
    /*@reldef@*/ STAILQ_ENTRY(dwarf2_loc) link;

    unsigned long vline;    /* virtual line number of .loc directive */

    /* source information */
    unsigned long file;     /* index into table of filenames */
    unsigned long line;     /* source line number */
    unsigned long column;   /* source column */
    unsigned long discriminator;
    int isa_change;
    unsigned long isa;
    enum {
        IS_STMT_NOCHANGE = 0,
        IS_STMT_SET,
        IS_STMT_CLEAR
    } is_stmt;
    int basic_block;
    int prologue_end;
    int epilogue_begin;

    yasm_bytecode *bc;      /* first bytecode following */
    yasm_symrec *sym;       /* last symbol preceding */
} dwarf2_loc;

/* Per-section data */
typedef struct dwarf2_section_data {
    /* The locations set by the .loc directives in this section, in assembly
     * source order.
     */
    /*@reldef@*/ STAILQ_HEAD(dwarf2_lochead, dwarf2_loc) locs;
} dwarf2_section_data;

extern const yasm_assoc_data_callback yasm_dwarf2__section_data_cb;

yasm_bytecode *yasm_dwarf2__append_bc(yasm_section *sect, yasm_bytecode *bc);

/*@dependent@*/ yasm_symrec *yasm_dwarf2__bc_sym(yasm_symtab *symtab,
                                                 yasm_bytecode *bc);

typedef struct dwarf2_head dwarf2_head;
dwarf2_head *yasm_dwarf2__add_head
    (yasm_dbgfmt_dwarf2 *dbgfmt_dwarf2, yasm_section *sect,
     /*@null@*/ yasm_section *debug_ptr, int with_address, int with_segment);
void yasm_dwarf2__set_head_end(dwarf2_head *head, yasm_bytecode *end_prevbc);

/* Line number functions */
yasm_section *yasm_dwarf2__generate_line
    (yasm_object *object, yasm_linemap *linemap, yasm_errwarns *errwarns,
     int asm_source, /*@out@*/ yasm_section **main_code,
     /*@out@*/ size_t *num_line_sections);
void yasm_dwarf2__dir_loc(yasm_object *object, yasm_valparamhead *valparams,
                          yasm_valparamhead *objext_valparams,
                          unsigned long line);
void yasm_dwarf2__dir_file(yasm_object *object, yasm_valparamhead *valparams,
                           yasm_valparamhead *objext_valparams,
                           unsigned long line);

/* Address range table functions */
yasm_section *yasm_dwarf2__generate_aranges(yasm_object *object,
                                            yasm_section *debug_info);

/* Name lookup table functions */
yasm_section *yasm_dwarf2__generate_pubnames(yasm_object *object,
                                             yasm_section *debug_info);

/* Information functions */
yasm_section *yasm_dwarf2__generate_info
    (yasm_object *object, yasm_section *debug_line,
     /*@null@*/ yasm_section *main_code);

#endif
