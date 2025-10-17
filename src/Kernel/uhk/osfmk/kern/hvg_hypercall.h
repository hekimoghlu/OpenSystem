/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 7, 2023.
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
#ifndef _KERN_HVG_HYPERCALL_H_
#define _KERN_HVG_HYPERCALL_H_

#include <os/base.h>
#include <stdint.h>
#include <stdbool.h>
#include <uuid/uuid.h>

/* Architecture-independent definitions (exported to userland) */

/*
 * Apple Hypercall arguments
 */
typedef struct hvg_hcall_args {
	uint64_t args[6];
} hvg_hcall_args_t;


/*
 * Apple Hypercall return output
 */
typedef struct hvg_hcall_output {
	uint64_t regs[7];
} hvg_hcall_output_t;


/*
 * Apple Hypercall return code
 */

OS_CLOSED_ENUM(hvg_hcall_return, uint32_t,
    HVG_HCALL_SUCCESS             = 0x0000,       /* The call succeeded */
    HVG_HCALL_ACCESS_DENIED       = 0x0001,       /* Invalid access right */
    HVG_HCALL_INVALID_CODE        = 0x0002,       /* Hypercall code not recognized */
    HVG_HCALL_INVALID_PARAMETER   = 0x0003,       /* Specified register value not valid */
    HVG_HCALL_IO_FAILED           = 0x0004,       /* Input/output error */
    HVG_HCALL_FEAT_DISABLED       = 0x0005,       /* Feature not available */
    HVG_HCALL_UNSUPPORTED         = 0x0006,       /* Hypercall not supported */
    );


/*
 * Apple Hypercall call code
 */

OS_CLOSED_ENUM(hvg_hcall_code, uint32_t,
    HVG_HCALL_TRIGGER_DUMP        = 0x0001,       /* Collect guest dump */
    HVG_HCALL_SET_COREDUMP_DATA   = 0x0002,       /* Set necessary info for reliable coredump */
    HVG_HCALL_GET_MABS_OFFSET     = 0x0003,       /* Determine mach_absolute_time offset */
    HVG_HCALL_GET_BOOTSESSIONUUID = 0x0004,       /* Read host boot sesssion ID */
    HVG_HCALL_VCPU_WFK            = 0x0005,       /* Wait-for-kick (or exception) */
    HVG_HCALL_VCPU_KICK           = 0x0006,       /* Kick a vcpu */
    HVG_HCALL_COUNT
    );

/*
 * Options for collecting kernel vmcore
 */

OS_CLOSED_OPTIONS(hvg_hcall_dump_option, uint32_t,
    HVG_HCALL_DUMP_OPTION_REGULAR   =  0x0001     /* Regular dump-guest-memory */
    );

typedef struct hvg_hcall_vmcore_file {
	char tag[57];   /* 7 64-bit registers plus 1 byte for '\0' */
} hvg_hcall_vmcore_file_t;


#ifdef XNU_KERNEL_PRIVATE

/*
 * For XNU kernel use only (omitted from userland headers)
 */

extern bool hvg_is_hcall_available(hvg_hcall_code_t);

extern void hvg_hcall_set_coredump_data(void);

extern hvg_hcall_return_t hvg_hcall_trigger_dump(hvg_hcall_vmcore_file_t *vmcore,
    const hvg_hcall_dump_option_t dump_option);
extern hvg_hcall_return_t hvg_hcall_get_bootsessionuuid(uuid_string_t uuid);
extern hvg_hcall_return_t hvg_hcall_get_mabs_offset(uint64_t *mabs_offset);
extern void hvg_hc_kick_cpu(unsigned int cpu_id);
extern void hvg_hc_wait_for_kick(unsigned int ien);

#endif /* XNU_KERNEL_PRIVATE */

#endif /* _KERN_HV_HYPERCALL_H_ */
