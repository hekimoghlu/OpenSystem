/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 10, 2024.
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
 *	File: libkern/kernel_mach_header.h
 *
 *	Definitions for accessing mach-o headers.
 *
 * NOTE:	These functions work on Mach-O headers compatible with
 *		the currently running kernel, and cannot be used against mach
 *		headers other than that of the currently running kernel.
 *
 */

#ifndef _KERNEL_MACH_HEADER_
#define _KERNEL_MACH_HEADER_

#ifdef __cplusplus
extern "C" {
#endif

#include <mach/mach_types.h>
#include <mach-o/loader.h>
#include <mach-o/nlist.h>
#include <mach-o/reloc.h>

#if     !KERNEL
#error this header for kernel use only
#endif

#if defined(__LP64__)

typedef struct mach_header_64   kernel_mach_header_t;
typedef struct segment_command_64 kernel_segment_command_t;
typedef struct section_64               kernel_section_t;
typedef struct nlist_64         kernel_nlist_t;

#define MH_MAGIC_KERNEL         MH_MAGIC_64
#define LC_SEGMENT_KERNEL       LC_SEGMENT_64

#else

typedef struct mach_header              kernel_mach_header_t;
typedef struct segment_command  kernel_segment_command_t;
typedef struct section                  kernel_section_t;
typedef struct nlist            kernel_nlist_t;

#define MH_MAGIC_KERNEL         MH_MAGIC
#define LC_SEGMENT_KERNEL               LC_SEGMENT
#define SECT_CONSTRUCTOR                "__constructor"
#define SECT_DESTRUCTOR                 "__destructor"

#endif

#define SECT_MODINITFUNC                "__mod_init_func"
#define SECT_MODTERMFUNC                "__mod_term_func"

extern kernel_mach_header_t _mh_execute_header;

/*
 * If the 'MH_DYLIB_IN_CACHE' bit is set in a kernel or kext mach-o header flag,
 * then that mach-o has been linked by the new KernelCollectionBuilder into
 * an MH_FILESET kernel collection. This bit is typically reserved for dylibs
 * that are part of the dyld-shared-cache, but when applied to constituents of
 * a kernel collection, it has this special meaning.
 */
#define kernel_mach_header_is_in_fileset(_mh) ((_mh)->flags & MH_DYLIB_IN_CACHE)

vm_offset_t getlastaddr(kernel_mach_header_t *header);
vm_offset_t getlastkerneladdr(void);

kernel_segment_command_t *firstseg(void);
kernel_segment_command_t *firstsegfromheader(kernel_mach_header_t *header);
kernel_segment_command_t *nextsegfromheader(
	kernel_mach_header_t    *header,
	kernel_segment_command_t        *seg);
kernel_segment_command_t *getsegbyname(const char *seg_name);
kernel_segment_command_t *getsegbynamefromheader(
	kernel_mach_header_t    *header,
	const char              *seg_name);
void *getsegdatafromheader(kernel_mach_header_t *, const char *, unsigned long *);
kernel_section_t *getsectbyname(const char *seg_name, const char *sect_name);
kernel_section_t *getsectbynamefromheader(
	kernel_mach_header_t    *header,
	const char              *seg_name,
	const char              *sect_name);
kernel_section_t *getsectbynamefromseg(
	kernel_segment_command_t        *sgp,
	const char                      *segname,
	const char                      *sectname);
uint32_t getsectoffsetfromheader(
	kernel_mach_header_t *mhp,
	const char *segname,
	const char *sectname);
void *getsectdatafromheader(kernel_mach_header_t *, const char *, const char *, unsigned long *);
kernel_section_t *firstsect(kernel_segment_command_t *sgp);
kernel_section_t *nextsect(kernel_segment_command_t *sgp, kernel_section_t *sp);
void *getcommandfromheader(kernel_mach_header_t *, uint32_t);
void *getuuidfromheader(kernel_mach_header_t *, unsigned long *);

bool kernel_text_contains(vm_offset_t);

#ifdef __cplusplus
}
#endif

#endif  /* _KERNEL_MACH_HEADER_ */
