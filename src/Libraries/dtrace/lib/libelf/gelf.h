/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 27, 2024.
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
 * Copyright 2004 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

#ifndef	_GELF_H
#define	_GELF_H

#ifndef _DARWIN_SHIM_H
#include <stdint.h>
typedef uint8_t		uchar_t;
typedef uint16_t	ushort_t;
typedef uint32_t	uint_t;
typedef unsigned long	ulong_t;
typedef uint64_t	u_longlong_t;
typedef int64_t		longlong_t;
typedef int64_t		off64_t;

#endif

#include <libelf.h>

#ifdef	__cplusplus
extern "C" {
#endif

/*
 * Class-independent ELF API for Elf utilities.  This is
 * for manipulating Elf32 and Elf64 specific information
 * in a format common to both classes.
 */

typedef Elf64_Addr	GElf_Addr;
typedef Elf64_Half	GElf_Half;
typedef Elf64_Sxword	GElf_Sxword;
typedef Elf64_Word	GElf_Word;
typedef Elf64_Xword	GElf_Xword;
typedef Elf64_Ehdr	GElf_Ehdr;
typedef Elf64_Shdr	GElf_Shdr;
typedef Elf64_Sym	GElf_Sym;

/*
 * sym.st_info field is same size for Elf32 and Elf64.
 */
#define	GELF_ST_BIND	ELF64_ST_BIND
#define	GELF_ST_TYPE	ELF64_ST_TYPE
#define	GELF_ST_INFO	ELF64_ST_INFO

int		gelf_getclass(Elf*);
GElf_Ehdr *	gelf_getehdr(Elf *, GElf_Ehdr *);
GElf_Shdr *	gelf_getshdr(Elf_Scn *,  GElf_Shdr *);
GElf_Sym *	gelf_getsym(Elf_Data *, int, GElf_Sym *);


#ifdef	__cplusplus
}
#endif

#endif	/* _GELF_H */
