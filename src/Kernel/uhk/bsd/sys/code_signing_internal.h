/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 25, 2022.
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
#ifndef _SYS_CODE_SIGNING_INTERNAL_H_
#define _SYS_CODE_SIGNING_INTERNAL_H_

#include <sys/cdefs.h>
__BEGIN_DECLS

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnullability-completeness"
#pragma GCC diagnostic ignored "-Wnullability-completeness-on-arrays"

#ifdef XNU_KERNEL_PRIVATE

#include <mach/boolean.h>
#include <mach/kern_return.h>
#include <kern/cs_blobs.h>
#include <vm/pmap.h>
#include <vm/pmap_cs.h>
#include <img4/firmware.h>
#include <libkern/image4/dlxk.h>

#if CONFIG_SPTM
/* TrustedExecutionMonitor */
#define CODE_SIGNING_MONITOR 1
#define CODE_SIGNING_MONITOR_PREFIX txm

#elif PMAP_CS_PPL_MONITOR
/* Page Protection Layer -- PMAP_CS */
#define CODE_SIGNING_MONITOR 1
#define CODE_SIGNING_MONITOR_PREFIX ppl

#else
/* No monitor -- XNU */
#define CODE_SIGNING_MONITOR 0
#define CODE_SIGNING_MONITOR_PREFIX xnu

#endif /* CONFIG_SPTM */

/**
 * This macro can be used by code which is abstracting out the concept of the code
 * signing monitor in order to redirect calls to the correct monitor environment.
 */
#define __CSM_PREFIX(prefix, name) prefix##_##name
#define _CSM_PREFIX(prefix, name)  __CSM_PREFIX(prefix, name)
#define CSM_PREFIX(name)           _CSM_PREFIX(CODE_SIGNING_MONITOR_PREFIX, name)

void CSM_PREFIX(toggle_developer_mode)(
	bool state);

kern_return_t CSM_PREFIX(rem_enable)(void);

kern_return_t CSM_PREFIX(rem_state)(void);

kern_return_t CSM_PREFIX(secure_channel_shared_page)(
	uint64_t * secure_channel_phys,
	size_t *secure_channel_size);

void CSM_PREFIX(update_device_state)(void);

void CSM_PREFIX(complete_security_boot_mode)(
	uint32_t security_boot_mode);

void CSM_PREFIX(set_compilation_service_cdhash)(
	const uint8_t cdhash[CS_CDHASH_LEN]);

bool CSM_PREFIX(match_compilation_service_cdhash)(
	const uint8_t cdhash[CS_CDHASH_LEN]);

void CSM_PREFIX(set_local_signing_public_key)(
	const uint8_t * public_key);

uint8_t* CSM_PREFIX(get_local_signing_public_key)(void);

void* CSM_PREFIX(image4_storage_data)(
	size_t * allocated_size);

void CSM_PREFIX(image4_set_nonce)(
	const img4_nonce_domain_index_t ndi,
	const img4_nonce_t *nonce);

void CSM_PREFIX(image4_roll_nonce)(
	const img4_nonce_domain_index_t ndi);

errno_t CSM_PREFIX(image4_copy_nonce)(
	const img4_nonce_domain_index_t ndi,
	img4_nonce_t *nonce_out);

errno_t CSM_PREFIX(image4_execute_object)(
	img4_runtime_object_spec_index_t obj_spec_index,
	const img4_buff_t *payload,
	const img4_buff_t *manifest);

errno_t CSM_PREFIX(image4_copy_object)(
	img4_runtime_object_spec_index_t obj_spec_index,
	vm_address_t object_out,
	size_t *object_length);

const void* CSM_PREFIX(image4_get_monitor_exports)(void);

errno_t CSM_PREFIX(image4_set_release_type)(
	const char *release_type);

errno_t CSM_PREFIX(image4_set_bnch_shadow)(
	const img4_nonce_domain_index_t ndi);

kern_return_t CSM_PREFIX(image4_transfer_region)(
	image4_cs_trap_t selector,
	vm_address_t region_addr,
	vm_size_t region_size);

kern_return_t CSM_PREFIX(image4_reclaim_region)(
	image4_cs_trap_t selector,
	vm_address_t region_addr,
	vm_size_t region_size);

errno_t CSM_PREFIX(image4_monitor_trap)(
	image4_cs_trap_t selector,
	const void *input_data,
	size_t input_size);

#if CODE_SIGNING_MONITOR
/* Function prototypes needed only when we have a monitor environment */

bool CSM_PREFIX(code_signing_enabled)(void);

void CSM_PREFIX(enter_lockdown_mode)(void);

vm_size_t CSM_PREFIX(managed_code_signature_size)(void);

void CSM_PREFIX(unrestrict_local_signing_cdhash)(
	const uint8_t cdhash[CS_CDHASH_LEN]);

kern_return_t CSM_PREFIX(register_provisioning_profile)(
	const void *profile_blob,
	const size_t profile_blob_size,
	void **profile_obj);

kern_return_t CSM_PREFIX(trust_provisioning_profile)(
	void *profile_obj,
	const void *sig_data,
	size_t sig_size);

kern_return_t CSM_PREFIX(unregister_provisioning_profile)(
	void *profile_obj);

kern_return_t CSM_PREFIX(associate_provisioning_profile)(
	void *sig_obj,
	void *profile_obj);

kern_return_t CSM_PREFIX(disassociate_provisioning_profile)(
	void *sig_obj);

kern_return_t CSM_PREFIX(register_code_signature)(
	const vm_address_t signature_addr,
	const vm_size_t signature_size,
	const vm_offset_t code_directory_offset,
	const char *signature_path,
	void **sig_obj,
	vm_address_t *txm_signature_addr);

kern_return_t CSM_PREFIX(unregister_code_signature)(
	void *sig_obj);

kern_return_t CSM_PREFIX(verify_code_signature)(
	void *sig_obj,
	uint32_t *trust_level);

kern_return_t CSM_PREFIX(reconstitute_code_signature)(
	void *sig,
	vm_address_t *unneeded_addr,
	vm_size_t *unneeded_size);

kern_return_t CSM_PREFIX(associate_code_signature)(
	pmap_t pmap,
	void *sig_obj,
	const vm_address_t region_addr,
	const vm_size_t region_size,
	const vm_offset_t region_offset);

kern_return_t CSM_PREFIX(allow_jit_region)(
	pmap_t pmap);

kern_return_t CSM_PREFIX(associate_jit_region)(
	pmap_t pmap,
	const vm_address_t region_addr,
	const vm_size_t region_size);

kern_return_t CSM_PREFIX(associate_debug_region)(
	pmap_t pmap,
	const vm_address_t region_addr,
	const vm_size_t region_size);

kern_return_t CSM_PREFIX(address_space_debugged)(
	pmap_t pmap);

kern_return_t CSM_PREFIX(allow_invalid_code)(
	pmap_t pmap);

kern_return_t CSM_PREFIX(get_trust_level_kdp)(
	pmap_t pmap,
	uint32_t *trust_level);

kern_return_t CSM_PREFIX(get_jit_address_range_kdp)(
	pmap_t pmap,
	uintptr_t *jit_region_start,
	uintptr_t *jit_region_end);

kern_return_t CSM_PREFIX(address_space_exempt)(
	const pmap_t pmap);

kern_return_t CSM_PREFIX(fork_prepare)(
	pmap_t old_pmap,
	pmap_t new_pmap);

kern_return_t CSM_PREFIX(acquire_signing_identifier)(
	const void *sig_obj,
	const char **signing_id);

kern_return_t CSM_PREFIX(associate_kernel_entitlements)(
	void *sig_obj,
	const void *kernel_entitlements);

kern_return_t CSM_PREFIX(resolve_kernel_entitlements)(
	pmap_t pmap,
	const void **kernel_entitlements);

kern_return_t CSM_PREFIX(accelerate_entitlements)(
	void *sig_obj,
	CEQueryContext_t *ce_ctx);

#endif /* CODE_SIGNING_MONITOR */

#endif /* XNU_KERNEL_PRIVATE */

#pragma GCC diagnostic pop

__END_DECLS
#endif /* _SYS_CODE_SIGNING_INTERNAL_H_ */
