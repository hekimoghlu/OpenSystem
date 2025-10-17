/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 1, 2024.
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
#ifndef _KERNEL_TELEMETRY_H_
#define _KERNEL_TELEMETRY_H_

#include <stdint.h>
#include <sys/cdefs.h>
#include <mach/mach_types.h>
#include <kern/thread.h>

__BEGIN_DECLS

/*
 * No longer supported.
 */
#define TELEMETRY_CMD_TIMER_EVENT 1
#define TELEMETRY_CMD_VOUCHER_NAME 2
#define TELEMETRY_CMD_VOUCHER_STAIN TELEMETRY_CMD_VOUCHER_NAME

enum telemetry_pmi {
	TELEMETRY_PMI_NONE,
	TELEMETRY_PMI_INSTRS,
	TELEMETRY_PMI_CYCLES,
};
#define TELEMETRY_CMD_PMI_SETUP 3

#if XNU_KERNEL_PRIVATE

/* implemented in OSKextLib.cpp */
extern void telemetry_backtrace_add_kexts(
	char                 *buf,
	size_t                buflen,
	uintptr_t            *frames,
	uint32_t              framecnt);

extern void telemetry_backtrace_to_string(
	char                 *buf,
	size_t                buflen,
	uint32_t              tot,
	uintptr_t            *frames);

extern void telemetry_init(void);

extern void compute_telemetry(void *);

extern void telemetry_ast(thread_t thread, uint32_t reasons);

extern int telemetry_kernel_gather(user_addr_t user_buffer, uint32_t *user_length);
extern int telemetry_gather(user_addr_t buffer, uint32_t *length, bool mark);

extern int telemetry_pmi_setup(enum telemetry_pmi pmi_type, uint64_t interval);

#if CONFIG_MACF
extern int telemetry_macf_mark_curthread(void);
#endif

extern void bootprofile_wake_from_sleep(void);
extern void bootprofile_get(void **buffer, uint32_t *length);
extern int bootprofile_gather(user_addr_t buffer, uint32_t *length);

#endif /* XNU_KERNEL_PRIVATE */

__END_DECLS

#endif /* _KERNEL_TELEMETRY_H_ */
