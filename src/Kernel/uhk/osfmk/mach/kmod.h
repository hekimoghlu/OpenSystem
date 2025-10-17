/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 26, 2023.
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
 * NOTICE: This file was modified by SPARTA, Inc. in 2005 to introduce
 * support for mandatory and extensible security protections.  This notice
 * is included in support of clause 2.2 (b) of the Apple Public License,
 * Version 2.0.
 */

#ifndef    _MACH_KMOD_H_
#define    _MACH_KMOD_H_

#include <mach/kern_return.h>
#include <mach/mach_types.h>

#include <sys/cdefs.h>

__BEGIN_DECLS

#if PRAGMA_MARK
#pragma mark Basic macros & typedefs
#endif
/***********************************************************************
* Basic macros & typedefs
***********************************************************************/
#define KMOD_MAX_NAME    64

#define KMOD_RETURN_SUCCESS    KERN_SUCCESS
#define KMOD_RETURN_FAILURE    KERN_FAILURE

typedef int kmod_t;

struct  kmod_info;
typedef kern_return_t kmod_start_func_t(struct kmod_info * ki, void * data);
typedef kern_return_t kmod_stop_func_t(struct kmod_info * ki, void * data);

#if PRAGMA_MARK
#pragma mark Structure definitions
#endif
/***********************************************************************
* Structure definitions
*
* All structures must be #pragma pack(4).
***********************************************************************/
#pragma pack(push, 4)

/* Run-time struct only; never saved to a file */
typedef struct kmod_reference {
	struct kmod_reference * next;
	struct kmod_info      * info;
} kmod_reference_t;

/***********************************************************************
* Warning: Any changes to the kmod_info structure affect the
* KMOD_..._DECL macros below.
***********************************************************************/

/* The kmod_info_t structure is only safe to use inside the running
 * kernel.  If you need to work with a kmod_info_t structure outside
 * the kernel, please use the compatibility definitions below.
 */
typedef struct kmod_info {
	struct kmod_info  * next;
	int32_t             info_version;       // version of this structure
	uint32_t            id;
	char                name[KMOD_MAX_NAME];
	char                version[KMOD_MAX_NAME];
	int32_t             reference_count;    // # linkage refs to this
	kmod_reference_t  * reference_list;     // who this refs (links on)
	vm_address_t        address;            // starting address
	vm_size_t           size;               // total size
	vm_size_t           hdr_size;           // unwired hdr size
	kmod_start_func_t * start;
	kmod_stop_func_t  * stop;
} kmod_info_t;

/* A compatibility definition of kmod_info_t for 32-bit kexts.
 */
typedef struct kmod_info_32_v1 {
	uint32_t            next_addr;
	int32_t             info_version;
	uint32_t            id;
	uint8_t             name[KMOD_MAX_NAME];
	uint8_t             version[KMOD_MAX_NAME];
	int32_t             reference_count;
	uint32_t            reference_list_addr;
	uint32_t            address;
	uint32_t            size;
	uint32_t            hdr_size;
	uint32_t            start_addr;
	uint32_t            stop_addr;
} kmod_info_32_v1_t;

/* A compatibility definition of kmod_info_t for 64-bit kexts.
 */
typedef struct kmod_info_64_v1 {
	uint64_t            next_addr;
	int32_t             info_version;
	uint32_t            id;
	uint8_t             name[KMOD_MAX_NAME];
	uint8_t             version[KMOD_MAX_NAME];
	int32_t             reference_count;
	uint64_t            reference_list_addr;
	uint64_t            address;
	uint64_t            size;
	uint64_t            hdr_size;
	uint64_t            start_addr;
	uint64_t            stop_addr;
} kmod_info_64_v1_t;

#pragma pack(pop)

#if PRAGMA_MARK
#pragma mark Kmod structure declaration macros
#endif
/***********************************************************************
* Kmod structure declaration macros
***********************************************************************/
#define KMOD_INFO_NAME       kmod_info
#define KMOD_INFO_VERSION    1

#define KMOD_DECL(name, version)                                  \
    static kmod_start_func_t name ## _module_start;               \
    static kmod_stop_func_t  name ## _module_stop;                \
    kmod_info_t KMOD_INFO_NAME = { 0, KMOD_INFO_VERSION, -1U,      \
	               { #name }, { version }, -1, 0, 0, 0, 0,    \
	                   name ## _module_start,                 \
	                   name ## _module_stop };

#define KMOD_EXPLICIT_DECL(name, version, start, stop)            \
    kmod_info_t KMOD_INFO_NAME = { 0, KMOD_INFO_VERSION, -1U,      \
	               { #name }, { version }, -1, 0, 0, 0, 0,    \
	                   start, stop };

#if PRAGMA_MARK
#pragma mark Kernel private declarations
#endif
/***********************************************************************
* Kernel private declarations.
***********************************************************************/
#ifdef    KERNEL_PRIVATE

/* Implementation now in libkern/OSKextLib.cpp. */
extern void kmod_panic_dump(vm_offset_t * addr, unsigned int dump_cnt);

#if CONFIG_DTRACE
/*
 * DTrace can take a flag indicating whether it should instrument
 * probes immediately based on kernel symbols.  This per kext
 * flag overrides system mode in dtrace_modload().
 */
#define KMOD_DTRACE_FORCE_INIT  0x01
#define KMOD_DTRACE_STATIC_KEXT 0x02
#define KMOD_DTRACE_NO_KERNEL_SYMS 0x04
#endif /* CONFIG_DTRACE */

#endif    /* KERNEL_PRIVATE */


#if PRAGMA_MARK
#pragma mark Obsolete kmod stuff
#endif
/***********************************************************************
* These 3 should be dropped but they're referenced by MIG declarations.
***********************************************************************/
typedef void * kmod_args_t;
typedef int kmod_control_flavor_t;
typedef kmod_info_t * kmod_info_array_t;

__END_DECLS

#endif    /* _MACH_KMOD_H_ */
