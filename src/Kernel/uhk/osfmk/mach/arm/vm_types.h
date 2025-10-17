/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 12, 2022.
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
 * Copyright (c) 1991,1990,1989,1988 Carnegie Mellon University
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
 *	File:	vm_types.h
 *	Author:	Avadis Tevanian, Jr.
 *	Date: 1985
 *
 *	Header file for VM data types.  ARM version.
 */

#ifndef _MACH_ARM_VM_TYPES_H_
#define _MACH_ARM_VM_TYPES_H_

#if defined (__arm__) || defined (__arm64__)

#ifndef ASSEMBLER

#include <arm/_types.h>
#include <stdint.h>
#include <Availability.h>
#include <sys/cdefs.h>

/*
 * natural_t and integer_t are Mach's legacy types for machine-
 * independent integer types (unsigned, and signed, respectively).
 * Their original purpose was to define other types in a machine/
 * compiler independent way.
 *
 * They also had an implicit "same size as pointer" characteristic
 * to them (i.e. Mach's traditional types are very ILP32 or ILP64
 * centric).  We will likely support x86 ABIs that do not follow
 * either ofthese models (specifically LP64).  Therefore, we had to
 * make a choice between making these types scale with pointers or stay
 * tied to integers.  Because their use is predominantly tied to
 * to the size of an integer, we are keeping that association and
 * breaking free from pointer size guarantees.
 *
 * New use of these types is discouraged.
 */
typedef __darwin_natural_t      natural_t;
typedef int                     integer_t;

/*
 * A vm_offset_t is a type-neutral pointer,
 * e.g. an offset into a virtual memory space.
 */
#ifdef __LP64__
typedef uintptr_t               vm_offset_t __kernel_ptr_semantics;
typedef uintptr_t               vm_size_t;

typedef uint64_t                mach_vm_address_t __kernel_ptr_semantics;
typedef uint64_t                mach_vm_offset_t __kernel_ptr_semantics;
typedef uint64_t                mach_vm_size_t;

typedef uint64_t                vm_map_offset_t __kernel_ptr_semantics;
typedef uint64_t                vm_map_address_t __kernel_ptr_semantics;
typedef uint64_t                vm_map_size_t;
#else
typedef natural_t               vm_offset_t __kernel_ptr_semantics;
/*
 * A vm_size_t is the proper type for e.g.
 * expressing the difference between two
 * vm_offset_t entities.
 */
typedef natural_t               vm_size_t;

/*
 * This new type is independent of a particular vm map's
 * implementation size - and represents appropriate types
 * for all possible maps.  This is used for interfaces
 * where the size of the map is not known - or we don't
 * want to have to distinguish.
 */
#if defined(__IPHONE_OS_VERSION_MIN_REQUIRED) && (__IPHONE_OS_VERSION_MIN_REQUIRED < __IPHONE_5_0)
typedef uint32_t                mach_vm_address_t;
typedef uint32_t                mach_vm_offset_t;
typedef uint32_t                mach_vm_size_t;
#else
typedef uint64_t                mach_vm_address_t __kernel_ptr_semantics;
typedef uint64_t                mach_vm_offset_t __kernel_ptr_semantics;
typedef uint64_t                mach_vm_size_t;
#endif

typedef uint32_t                vm_map_offset_t __kernel_ptr_semantics;
typedef uint32_t                vm_map_address_t __kernel_ptr_semantics;
typedef uint32_t                vm_map_size_t;
#endif /* __LP64__ */


typedef uint32_t                vm32_offset_t;
typedef uint32_t                vm32_address_t;
typedef uint32_t                vm32_size_t;

typedef vm_offset_t             mach_port_context_t;

#ifdef MACH_KERNEL_PRIVATE
typedef vm32_offset_t           mach_port_context32_t;
typedef mach_vm_offset_t        mach_port_context64_t;
#endif

#endif  /* ASSEMBLER */

/*
 * If composing messages by hand (please do not)
 */
#define MACH_MSG_TYPE_INTEGER_T MACH_MSG_TYPE_INTEGER_32

#endif /* defined (__arm__) || defined (__arm64__) */

#endif  /* _MACH_ARM_VM_TYPES_H_ */
