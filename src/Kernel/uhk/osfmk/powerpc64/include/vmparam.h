/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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
#define	USRSTACK	VM_MAXUSER_ADDRESS

/*
 * Virtual memory related constants, all in bytes
 */
#define	MAXTSIZ		((paddr_t)256*1024*1024)	/* max text size */
#ifndef DFLDSIZ
#define	DFLDSIZ		((paddr_t)512*1024*1024)	/* initial data size limit */
#endif
#ifndef MAXDSIZ
#define	MAXDSIZ		((paddr_t)32*1024*1024*1024)	/* max data size */
#endif
#ifndef BRKSIZ
#define	BRKSIZ		((paddr_t)16*1024*1024*1024)	/* heap gap size */
#endif
#ifndef	DFLSSIZ
#define	DFLSSIZ		((paddr_t)2*1024*1024)		/* initial stack size limit */
#endif
#ifndef	MAXSSIZ
#define	MAXSSIZ		((paddr_t)32*1024*1024)		/* max stack size */
#endif

#define	STACKGAP_RANDOM	256*1024

/*
 * Size of shared memory map
 */
#ifndef	SHMMAXPGS
#define	SHMMAXPGS	1024
#endif

/*
 * Size of User Raw I/O map
 */
#define	USRIOSIZE 	300

#define	VM_PHYS_SIZE		(USRIOSIZE * PAGE_SIZE)

#define VM_PHYSSEG_MAX		32
#define VM_PHYSSEG_STRAT	VM_PSTRAT_BSEARCH

#define	VM_MIN_ADDRESS		((vaddr_t)PAGE_SIZE)
#define VM_MAXUSER_ADDRESS	0xbffffffffffff000UL
#define VM_MAX_ADDRESS		0xffffffffffffffffUL
#ifdef _KERNEL
#define VM_MIN_STACK_ADDRESS	0x9000000000000000UL
#endif
#define VM_MIN_KERNEL_ADDRESS	0xc000000000000000UL
#define VM_MAX_KERNEL_ADDRESS	0xc0000007ffffffffUL
