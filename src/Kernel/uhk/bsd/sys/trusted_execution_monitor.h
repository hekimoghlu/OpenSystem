/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 6, 2023.
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
#ifndef _SYS_CODE_SIGNING_TXM_H_
#define _SYS_CODE_SIGNING_TXM_H_

#if CONFIG_SPTM

#include <libkern/section_keywords.h>
#include <kern/locks.h>
#include <kern/lock_rw.h>
#include <vm/pmap.h>
#include <sys/queue.h>
#include <TrustedExecutionMonitor/API.h>

#ifndef kTXMImage4APIVersion
#define kTXMImage4APIVersion 0
#endif

/* These are hidden behind MACH_KERNEL_PRIVATE in other files */
typedef uint64_t pmap_paddr_t __kernel_ptr_semantics;
extern vm_map_address_t phystokv(pmap_paddr_t pa);
extern pmap_paddr_t kvtophys_nofail(vm_offset_t va);

/* Global read-only data of TXM */
extern const TXMReadOnlyData_t *txm_ro_data;

/* Code signing configuration of TXM */
extern const CSConfig_t *txm_cs_config;

/* All statistical data collected from TXM */
extern const TXMStatistics_t *txm_stats;

/* All static trust cache information collected from TXM */
extern uint32_t txm_static_trust_caches;
extern TCCapabilities_t static_trust_cache_capabilities0;
extern TCCapabilities_t static_trust_cache_capabilities1;

typedef struct _txm_thread_stack {
	/* Virtual mapping of the thread stack page */
	uintptr_t thread_stack_papt;

	/* Physical page used for the thread stack */
	uintptr_t thread_stack_phys;

	/* Pointer to the thread stack structure on the thread stack page */
	TXMThreadStack_t *thread_stack_data;

	/* Linkage for the singly-linked-list */
	SLIST_ENTRY(_txm_thread_stack) link;
} txm_thread_stack_t;

typedef struct _txm_call {
	/* Input arguments */
	TXMKernelSelector_t selector;
	TXMReturnCode_t failure_code_silent;
	bool failure_fatal;
	bool failure_silent;
	bool skip_logs;
	uint32_t num_input_args;
	uint32_t num_output_args;

	/* Output arguments */
	TXMReturn_t txm_ret;
	uint64_t num_return_words;
	uint64_t return_words[kTXMStackReturnWords];
} txm_call_t;

/**
 * The main function to use for calling into the TrustedExecutionMonitor. This
 * function handles all the bits required, including allocation/deallocation of
 * the thread stack pages, the CPU instructions required to reach TXM, and also
 * going through the TXM buffer and capturing any logs left by the monitor.
 */
kern_return_t
txm_kernel_call(
	txm_call_t *parameters, ...);

/**
 * Go through the TrustedExecutionMonitor logging buffer and print all the logs
 * which TXM has added to it since the kernel last looked.
 */
void
txm_print_logs(void);

/**
 * Pages which need to be locked down by the TrustedExecutionMonitor need to made
 * owned by TXM. This function can be used to go through each physical page in a
 * range and transfer it to the relevant TXM type.
 */
void
txm_transfer_region(
	vm_address_t addr,
	vm_size_t size);

/**
 * As part of transferring a page to the TrustedExecutionMonitor, the range of
 * memory is always made read-only. This function can be used to go through all
 * of the mappings and make them read-write again. This can only be done when TXM
 * has transferred control of the pages back to the kernel.
 */
void
txm_reclaim_region(
	vm_address_t addr,
	vm_size_t size);

/**
 * Register an address space with the TrustedExecutionMonitor based on an address
 * space ID. This needs to be done AFTER the SPTM has made its call into TXM for
 * registering an address space ID otherwise the system will panic.
 */
kern_return_t
txm_register_address_space(
	pmap_t pmap,
	uint16_t addr_space_id,
	TXMAddressSpaceFlags_t flags);

/**
 * Unregister an address space from the TrustedExecutionMonitor using the address
 * space object which was previously returned from TXM. This needs to be done
 * AFTER the SPTM has unregistered the address space ID from TXM otherwise the
 * system will panic.
 */
kern_return_t
txm_unregister_address_space(
	pmap_t pmap);

#endif /* CONFIG_SPTM */
#endif /* _SYS_CODE_SIGNING_TXM_H_ */
