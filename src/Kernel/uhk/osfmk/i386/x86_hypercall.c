/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 6, 2022.
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
#include <kern/assert.h>
#include <kern/hvg_hypercall.h>
#include <i386/x86_hypercall.h>
#include <kdp/ml/i386/kdp_x86_common.h>
#include <i386/cpuid.h>
#include <os/log.h>
#include <vm/pmap.h>
#include <vm/vm_map_xnu.h>
#include <x86_64/lowglobals.h>

bool
hvg_is_hcall_available(hvg_hcall_code_t hcall)
{
	assert3u(hcall, <, HVG_HCALL_COUNT);

	const uint64_t features = cpuid_vmm_get_applepv_features();
	const uint64_t hcall_feature[HVG_HCALL_COUNT] = {
		[HVG_HCALL_TRIGGER_DUMP]      = CPUID_LEAF_FEATURE_COREDUMP,
		[HVG_HCALL_SET_COREDUMP_DATA] = CPUID_LEAF_FEATURE_XNU_DEBUG,
		[HVG_HCALL_GET_MABS_OFFSET]   = CPUID_LEAF_FEATURE_MABS_OFFSET,
		[HVG_HCALL_GET_BOOTSESSIONUUID] = CPUID_LEAF_FEATURE_BOOTSESSIONUUID,
	};
	return cpuid_vmm_present() && (features & hcall_feature[hcall]) != 0;
}

/*
 * This routine issues an Apple hypercall that notifies the hypervisor to
 * take a guest kernel coredump. If the vmcore argument is not NULL, the
 * name tag of the vmcore file is copied into the caller's vmcore tag array.
 * Otherwise the name tag is ignored.
 */

hvg_hcall_return_t
hvg_hcall_trigger_dump(hvg_hcall_vmcore_file_t *vmcore,
    const hvg_hcall_dump_option_t dump_option)
{
	assert(hvg_is_hcall_available(HVG_HCALL_TRIGGER_DUMP));
	assert3u(dump_option, ==, HVG_HCALL_DUMP_OPTION_REGULAR); /* Only known option for now. */

	hvg_hcall_output_regs_t output = {};
	const hvg_hcall_return_t ret = hvg_hypercall1(HVG_HCALL_TRIGGER_DUMP,
	    dump_option,
	    &output);

	if (ret != HVG_HCALL_SUCCESS) {
		return ret;
	}

	if (vmcore) {
		/* Caller requested vmcore tag to be returned */
		static_assert(sizeof(vmcore->tag) > sizeof(output), "not enough room for tag");
		static_assert(sizeof(vmcore->tag[0] * sizeof(uint64_t)) == sizeof(output.rax), "mis-match of tag and output sizes");

		const size_t reg_size = sizeof(uint64_t);

		memcpy(&vmcore->tag[reg_size * 0], &output.rax, reg_size);
		memcpy(&vmcore->tag[reg_size * 1], &output.rdi, reg_size);
		memcpy(&vmcore->tag[reg_size * 2], &output.rsi, reg_size);
		memcpy(&vmcore->tag[reg_size * 3], &output.rdx, reg_size);
		memcpy(&vmcore->tag[reg_size * 4], &output.rcx, reg_size);
		memcpy(&vmcore->tag[reg_size * 5], &output.r8, reg_size);
		memcpy(&vmcore->tag[reg_size * 6], &output.r9, reg_size);
		vmcore->tag[reg_size * 7] = '\0';
	}

	return HVG_HCALL_SUCCESS;
}

extern vm_offset_t c_buffers;
extern vm_size_t   c_buffers_size;

/*
 * Inform the hypervisor of the kernel physical address of
 * the low globals data and kernel CR3 value.
 */
void
hvg_hcall_set_coredump_data(void)
{
	assert(hvg_is_hcall_available(HVG_HCALL_SET_COREDUMP_DATA));

	hvg_hcall_output_regs_t output = {};

	/* Hypercall to set up necessary information for reliable coredump */
	const hvg_hcall_return_t ret = hvg_hypercall6(HVG_HCALL_SET_COREDUMP_DATA,
	    lowGlo.lgStext,              /* args[0]: KVA of kernel text */
	    kernel_map->min_offset,      /* args[1]: KVA of kernel_map_start */
	    kernel_map->max_offset,      /* args[2]: KVA of kernel_map_end */
	    kernel_pmap->pm_cr3,         /* args[3]: Kernel CR3 */
	    c_buffers,                   /* args[4]: KVA of compressor buffers */
	    c_buffers_size,              /* args[5]: Size of compressor buffers */
	    &output);

	if (ret != HVG_HCALL_SUCCESS) {
		os_log_error(OS_LOG_DEFAULT, "%s: hcall failed, ret %d\n",
		    __func__, ret);
	}
}

hvg_hcall_return_t
hvg_hcall_get_mabs_offset(uint64_t *mabs_offset)
{
	assert(hvg_is_hcall_available(HVG_HCALL_GET_MABS_OFFSET));

	hvg_hcall_output_regs_t output = {};

	const hvg_hcall_return_t ret = hvg_hypercall2(HVG_HCALL_GET_MABS_OFFSET,
	    pal_rtc_nanotime_info.tsc_base, pal_rtc_nanotime_info.ns_base,
	    &output);
	if (ret != HVG_HCALL_SUCCESS) {
		return ret;
	}

	*mabs_offset = output.rdi;
	return HVG_HCALL_SUCCESS;
}

hvg_hcall_return_t
hvg_hcall_get_bootsessionuuid(uuid_string_t uuid)
{
	assert(hvg_is_hcall_available(HVG_HCALL_GET_BOOTSESSIONUUID));

	hvg_hcall_output_regs_t output = {};

	const hvg_hcall_return_t ret = hvg_hypercall0(
		HVG_HCALL_GET_BOOTSESSIONUUID, &output);
	if (ret != HVG_HCALL_SUCCESS) {
		return ret;
	}

	static_assert(sizeof(uuid_string_t) == 37,
	    "unexpected uuid string length");

	memset(uuid, 0, sizeof(uuid_string_t));

	memcpy(&uuid[0], &output.rax, 8);
	memcpy(&uuid[8], &output.rdi, 8);
	memcpy(&uuid[16], &output.rsi, 8);
	memcpy(&uuid[24], &output.rdx, 8);
	memcpy(&uuid[32], &output.rcx, 4);

	return HVG_HCALL_SUCCESS;
}
