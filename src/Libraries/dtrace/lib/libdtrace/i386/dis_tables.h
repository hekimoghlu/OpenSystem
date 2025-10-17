/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 7, 2025.
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
/*
 * Copyright 2007 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

/*	Copyright (c) 1988 AT&T	*/
/*	  All Rights Reserved  	*/


#ifndef _DIS_TABLES_H
#define	_DIS_TABLES_H

/*
 * Constants and prototypes for the IA32 disassembler backend.  See dis_tables.c
 * for usage information and documentation.
 */

/*
 * APPLE NOTE: There is a copy of this file in kernel in
 * xnu:bsd/dev/i386/dis_tables.c
 *
 * It needs to be in sync with this file.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <sys/types.h>
#include "darwin_shim.h"
#include <sys/param.h>

/*
 * values for cpu mode
 */
#define	SIZE16	1
#define	SIZE32	2
#define	SIZE64	3

#define	OPLEN	256
#define	PFIXLEN	  8
#define	NCPS	12	/* number of chars per symbol	*/

/*
 * data structures that must be provided to dtrace_dis86()
 */
typedef struct d86opnd {
	char		d86_opnd[OPLEN];	/* symbolic rep of operand */
	char		d86_prefix[PFIXLEN];	/* any prefix string or "" */
	uint_t		d86_mode;		/* mode for immediate */
	uint_t		d86_value_size;		/* size in bytes of d86_value */
	uint64_t	d86_value;		/* immediate value of opnd */
} d86opnd_t;

typedef struct dis86 {
	uint_t		d86_mode;
	uint_t		d86_error;
	uint_t		d86_len;		/* instruction length */
	int		d86_rmindex;		/* index of modrm byte or -1 */
	uint_t		d86_memsize;		/* size of memory referenced */
	char		d86_bytes[16];		/* bytes of instruction */
	char		d86_mnem[OPLEN];
	uint_t		d86_numopnds;
	uint_t		d86_rex_prefix;		/* value of REX prefix if !0 */
	char		*d86_seg_prefix;	/* segment prefix, if any */
	uint_t		d86_opnd_size;
	uint_t		d86_addr_size;
	uint_t		d86_got_modrm;
	uint_t		d86_vsib;
	struct d86opnd	d86_opnd[4];		/* up to 4 operands */
	int		(*d86_check_func)(void *);
	int		(*d86_get_byte)(void *);
#ifdef DIS_TEXT
	int		(*d86_sym_lookup)(void *, uint64_t, char *, size_t);
	int		(*d86_sprintf_func)(char *, size_t, const char *, ...);
	int		d86_flags;
	uint_t		d86_imm_bytes;
#endif
	void		*d86_data;
} dis86_t;

extern int dtrace_disx86(dis86_t *x, uint_t cpu_mode);

#define	DIS_F_OCTAL	0x1	/* Print all numbers in octal */
#define	DIS_F_NOIMMSYM	0x2	/* Don't print symbols for immediates (.o) */

#ifdef DIS_TEXT
extern void dtrace_disx86_str(dis86_t *x, uint_t cpu_mode, uint64_t pc,
    char *buf, size_t len);
#endif

#ifdef __cplusplus
}
#endif

#endif /* _DIS_TABLES_H */
