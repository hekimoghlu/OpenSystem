/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 11, 2023.
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
#ifndef _NLIST_H_
#define	_NLIST_H_

/*
 * Symbol table entry format.  The #ifdef's are so that programs including
 * nlist.h can initialize nlist structures statically.
 */
struct nlist {
#ifdef _AOUT_INCLUDE_
	union {
		char *n_name;	/* symbol name (in memory) */
		long n_strx;	/* file string table offset (on disk) */
	} n_un;
#else
	char *n_name;		/* symbol name (in memory) */
#endif

#define	N_UNDF	0x00		/* undefined */
#define	N_ABS	0x02		/* absolute address */
#define	N_TEXT	0x04		/* text segment */
#define	N_DATA	0x06		/* data segment */
#define	N_BSS	0x08		/* bss segment */
#define	N_COMM	0x12		/* common reference */
#define	N_FN	0x1e		/* file name */

#define	N_EXT	0x01		/* external (global) bit, OR'ed in */
#define	N_TYPE	0x1e		/* mask for all the type bits */
	unsigned char n_type;	/* type defines */

	char n_other;		/* spare */
#define	n_hash	n_desc		/* used internally by ld(1); XXX */
	short n_desc;		/* used by stab entries */
	unsigned long n_value;	/* address/value of the symbol */
};

#define	N_FORMAT	"%08x"	/* namelist value format; XXX */
#define	N_STAB		0x0e0	/* mask for debugger symbols -- stab(5) */

#include <sys/cdefs.h>

__BEGIN_DECLS
int nlist(const char *, struct nlist *);
__END_DECLS

#endif /* !_NLIST_H_ */
