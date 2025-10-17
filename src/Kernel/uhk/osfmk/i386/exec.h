/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 5, 2025.
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
 * @OSF_COPYRIGHT@
 */
/*
 * Mach Operating System
 * Copyright (c) 1991,1990,1989,1988,1987 Carnegie Mellon University
 * All Rights Reserved.
 *
 * Permission to use, copy, modify and distribute this software and its
 * documentation is hereby granted, provided that both the copyright
 * notice and this permission notice appear in all copies of the
 * software, derivative works or modified versions, and any portions
 * thereof, and that both notices appear in supporting documentation.
 *
 * CARNEGIE MELLON ALLOWS FREE USE OF THIS SOFTWARE IN ITS "AS IS"
 * CONDITION.  CARNEGIE MELLON DISCLAIMS ANY LIABILITY OF ANY KIND FOR
 * ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
 *
 * Carnegie Mellon requests users of this software to return to
 *
 *  Software Distribution Coordinator  or  Software.Distribution@CS.CMU.EDU
 *  School of Computer Science
 *  Carnegie Mellon University
 *  Pittsburgh PA 15213-3890
 *
 * any improvements or extensions that they make and grant Carnegie Mellon
 * the rights to redistribute these changes.
 */
/*
 */
/*
 * exec stucture in an a.out file derived from FSF's
 * a.out.gnu.h file.
 */

#ifndef _EXEC_
#define _EXEC_  1

/*
 * Header prepended to each a.out file.
 */
struct exec {
#ifdef  sun
	unsigned short  a_machtype;/* machine type */
	unsigned short  a_info; /* Use macros N_MAGIC, etc for access */
#else   /* sun */
	unsigned long   a_info; /* Use macros N_MAGIC, etc for access */
#endif  /* sun */
	unsigned long a_text;   /* length of text, in bytes */
	unsigned long a_data;   /* length of data, in bytes */
	unsigned long a_bss;    /* length of uninitialized data area for file, in bytes */
	unsigned long a_syms;   /* length of symbol table data in file, in bytes */
	unsigned long a_entry;  /* start address */
	unsigned long a_trsize; /* length of relocation info for text, in bytes */
	unsigned long a_drsize; /* length of relocation info for data, in bytes */
};

/* Code indicating object file or impure executable.  */
#define OMAGIC 0407
/* Code indicating pure executable.  */
#define NMAGIC 0410
/* Code indicating demand-paged executable.  */
#define ZMAGIC 0413

#ifdef sun
/* Sun machine types */

#define M_OLDSUN2       0       /* old sun-2 executable files */
#define M_68010         1       /* runs on either 68010 or 68020 */
#define M_68020         2       /* runs only on 68020 */
#endif /* sun */

#endif  /* _EXEC_ */
