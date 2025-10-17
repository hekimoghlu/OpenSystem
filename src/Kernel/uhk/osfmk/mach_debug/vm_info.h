/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 18, 2024.
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
 * Copyright (c) 1991,1990 Carnegie Mellon University
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
 *	File:	mach_debug/vm_info.h
 *	Author:	Rich Draves
 *	Date:	March, 1990
 *
 *	Definitions for the VM debugging interface.
 */

#ifndef _MACH_DEBUG_VM_INFO_H_
#define _MACH_DEBUG_VM_INFO_H_

#include <mach/boolean.h>
#include <mach/machine/vm_types.h>
#include <mach/vm_inherit.h>
#include <mach/vm_prot.h>
#include <mach/memory_object_types.h>

#pragma pack(4)

/*
 *	Remember to update the mig type definitions
 *	in mach_debug_types.defs when adding/removing fields.
 */
typedef struct mach_vm_info_region {
	mach_vm_offset_t vir_start;     /* start of region */
	mach_vm_offset_t vir_end;       /* end of region */
	mach_vm_offset_t vir_object;    /* the mapped object(kernal addr) */
	memory_object_offset_t vir_offset;      /* offset into object */
	boolean_t vir_needs_copy;       /* does object need to be copied? */
	vm_prot_t vir_protection;       /* protection code */
	vm_prot_t vir_max_protection;   /* maximum protection */
	vm_inherit_t vir_inheritance;   /* inheritance */
	natural_t vir_wired_count;      /* number of times wired */
	natural_t vir_user_wired_count; /* number of times user has wired */
} mach_vm_info_region_t;

typedef struct vm_info_region_64 {
	natural_t vir_start;            /* start of region */
	natural_t vir_end;              /* end of region */
	natural_t vir_object;           /* the mapped object */
	memory_object_offset_t vir_offset;      /* offset into object */
	boolean_t vir_needs_copy;       /* does object need to be copied? */
	vm_prot_t vir_protection;       /* protection code */
	vm_prot_t vir_max_protection;   /* maximum protection */
	vm_inherit_t vir_inheritance;   /* inheritance */
	natural_t vir_wired_count;      /* number of times wired */
	natural_t vir_user_wired_count; /* number of times user has wired */
} vm_info_region_64_t;

typedef struct vm_info_region {
	natural_t vir_start;            /* start of region */
	natural_t vir_end;              /* end of region */
	natural_t vir_object;           /* the mapped object */
	natural_t vir_offset;           /* offset into object */
	boolean_t vir_needs_copy;       /* does object need to be copied? */
	vm_prot_t vir_protection;       /* protection code */
	vm_prot_t vir_max_protection;   /* maximum protection */
	vm_inherit_t vir_inheritance;   /* inheritance */
	natural_t vir_wired_count;      /* number of times wired */
	natural_t vir_user_wired_count; /* number of times user has wired */
} vm_info_region_t;


typedef struct vm_info_object {
	natural_t vio_object;           /* this object */
	natural_t vio_size;             /* object size (valid if internal - but too small) */
	unsigned int vio_ref_count;     /* number of references */
	unsigned int vio_resident_page_count; /* number of resident pages */
	unsigned int vio_absent_count;  /* number requested but not filled */
	natural_t vio_copy;             /* copy object */
	natural_t vio_shadow;           /* shadow object */
	natural_t vio_shadow_offset;    /* offset into shadow object */
	natural_t vio_paging_offset;    /* offset into memory object */
	memory_object_copy_strategy_t vio_copy_strategy;
	/* how to handle data copy */
	vm_offset_t vio_last_alloc;     /* offset of last allocation */
	/* many random attributes */
	unsigned int vio_paging_in_progress;
	boolean_t vio_pager_created;
	boolean_t vio_pager_initialized;
	boolean_t vio_pager_ready;
	boolean_t vio_can_persist;
	boolean_t vio_internal;
	boolean_t vio_temporary;
	boolean_t vio_alive;
	boolean_t vio_purgable;
	boolean_t vio_purgable_volatile;
} vm_info_object_t;

typedef vm_info_object_t *vm_info_object_array_t;

#pragma pack()

#endif  /* _MACH_DEBUG_VM_INFO_H_ */
