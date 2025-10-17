/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 4, 2024.
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
#include "niceprintf.h"


/* Definitions for the opcode table.  The table is indexed by the macros
   which are #defined in   defines.h   */

#define UNARY_OP 01
#define BINARY_OP 02

#define SPECIAL_FMT NULL

#define is_unary_op(x) (opcode_table[x].type == UNARY_OP)
#define is_binary_op(x) (opcode_table[x].type == BINARY_OP)
#define op_precedence(x) (opcode_table[x].prec)
#define op_format(x) (opcode_table[x].format)

/* _assoc_table -- encodes left-associativity and right-associativity
   information; indexed by precedence level.  Only 2, 3, 14 are
   right-associative.  Source:  Kernighan & Ritchie, p. 49 */

extern char _assoc_table[];

#define is_right_assoc(x) (_assoc_table [x])
#define is_left_assoc(x) (! _assoc_table [x])


typedef struct {
    int type;			/* UNARY_OP or BINARY_OP */
    int prec;			/* Precedence level, useful for adjusting
				   number of parens to insert.  Zero is a
				   special level, and 2, 3, 14 are
				   right-associative */
    char *format;
} table_entry;


extern char *fl_fmt_string;	/* Float constant format string */
extern char *db_fmt_string;	/* Double constant format string */
extern char *cm_fmt_string;	/* Complex constant format string */
extern char *dcm_fmt_string;	/* Double Complex constant format string */

extern int indent;		/* Number of spaces to indent; this is a
				   temporary fix */
extern int tab_size;		/* Number of spaces in each tab */
extern int in_string;

extern table_entry opcode_table[];


void	compgoto_out Argdcl((FILEP, tagptr, tagptr));
void	endif_out Argdcl((FILEP));
void	expr_out Argdcl((FILEP, tagptr));
void	out_and_free_statement Argdcl((FILEP, tagptr));
void	out_end_for Argdcl((FILEP));
void	out_if Argdcl((FILEP, tagptr));
void	out_name Argdcl((FILEP, Namep));
