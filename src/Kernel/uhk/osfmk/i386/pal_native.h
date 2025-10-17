/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 30, 2024.
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
#ifndef _I386_PAL_I386_H
#define _I386_PAL_I386_H

#ifdef XNU_KERNEL_PRIVATE

/* No-op on bare-metal */
#define pal_dbg_page_fault(x, y, z)
#define pal_dbg_set_task_name( x )
#define pal_set_signal_delivery( x )

#define pal_is_usable_memory(b, t)      (TRUE)

#define pal_hlt()                       __asm__ volatile ("sti; hlt")
#define pal_sti()                       __asm__ volatile ("sti")
#define pal_cli()                       __asm__ volatile ("cli")

static inline void
pal_stop_cpu(boolean_t cli)
{
	if (cli) {
		__asm__ volatile ( "cli");
	}
	__asm__ volatile ( "wbinvd; hlt");
}

#define pal_register_cache_state(t, v)

#define pal_execve_return(t)
#define pal_thread_terminate_self(t)
#define pal_ast_check(t)

#define panic_display_pal_info() do { } while(0)
#define pal_kernel_announce() do { } while(0)

#define PAL_AICPM_PROPERTY_VALUE 0

#define pal_pmc_swi() __asm__ __volatile__("int %0"::"i"(LAPIC_PMC_SWI_VECTOR):"memory")

/* Macro used by non-native xnus for access to low globals when it may
 * have moved.
 */
#define PAL_KDP_ADDR(x) (x)

struct pal_rtc_nanotime {
	volatile uint64_t       tsc_base;       /* timestamp */
	volatile uint64_t       ns_base;        /* nanoseconds */
	uint32_t                scale;          /* tsc -> nanosec multiplier */
	uint32_t                shift;          /* shift is nonzero only on "slow" machines, */
	                                        /* ie where tscFreq <= SLOW_TSC_THRESHOLD */
	volatile uint32_t       generation;     /* 0 == being updated */
	uint32_t                spare1;
};


#ifdef MACH_KERNEL_PRIVATE

struct pal_cpu_data {
};

struct pal_pcb {
};

struct pal_apic_table {
};

#endif /* MACH_KERNEL_PRIVATE */

#endif /* XNU_KERNEL_PRIVATE */

#endif /* _I386_PAL_I386_H */
