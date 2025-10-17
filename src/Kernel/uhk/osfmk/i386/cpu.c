/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 13, 2024.
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
 *	File:	i386/cpu.c
 *
 *	cpu specific routines
 */

#include <kern/misc_protos.h>
#include <kern/lock_group.h>
#include <kern/machine.h>
#include <mach/processor_info.h>
#include <i386/pmap.h>
#include <i386/machine_cpu.h>
#include <i386/machine_routines.h>
#include <i386/misc_protos.h>
#include <i386/cpu_threads.h>
#include <i386/rtclock_protos.h>
#include <i386/cpuid.h>
#include <i386/lbr.h>
#include <kern/debug.h>
#if CONFIG_VMX
#include <i386/vmx/vmx_cpu.h>
#endif
#include <vm/vm_kern.h>
#include <kern/timer_call.h>

const char *processor_to_datastring(const char *prefix, processor_t target_processor);

struct processor        processor_master;

/*ARGSUSED*/
kern_return_t
cpu_control(
	int                     slot_num,
	processor_info_t        info,
	unsigned int            count)
{
	printf("cpu_control(%d,%p,%d) not implemented\n",
	    slot_num, info, count);
	return KERN_FAILURE;
}

/*ARGSUSED*/
kern_return_t
cpu_info_count(
	__unused processor_flavor_t      flavor,
	unsigned int                    *count)
{
	*count = 0;
	return KERN_FAILURE;
}

/*ARGSUSED*/
kern_return_t
cpu_info(
	processor_flavor_t      flavor,
	int                     slot_num,
	processor_info_t        info,
	unsigned int            *count)
{
	printf("cpu_info(%d,%d,%p,%p) not implemented\n",
	    flavor, slot_num, info, count);
	return KERN_FAILURE;
}

void
cpu_sleep(void)
{
	cpu_data_t      *cdp = current_cpu_datap();

	/* This calls IOCPURunPlatformQuiesceActions when sleeping the boot cpu */
	PE_cpu_machine_quiesce(cdp->cpu_id);

	cpu_thread_halt();
}

void
cpu_init(void)
{
	cpu_data_t      *cdp = current_cpu_datap();

	timer_call_queue_init(&cdp->rtclock_timer.queue);
	cdp->rtclock_timer.deadline = EndOfAllTime;

	cdp->cpu_type = cpuid_cputype();
	cdp->cpu_subtype = cpuid_cpusubtype();

	i386_activate_cpu();
}

void
cpu_start(
	int cpu)
{
	kern_return_t           ret;

	if (cpu == cpu_number()) {
		cpu_machine_init();
		return;
	}

	/*
	 * Try to bring the CPU back online without a reset.
	 * If the fast restart doesn't succeed, fall back to
	 * the slow way.
	 */
	ret = intel_startCPU_fast(cpu);
	if (ret != KERN_SUCCESS) {
		/*
		 * Should call out through PE.
		 * But take the shortcut here.
		 */
		ret = intel_startCPU(cpu);
	}

	if (ret != KERN_SUCCESS) {
		panic("cpu_start(%d) failed: %d\n", cpu, ret);
	}
}

void
cpu_exit_wait(
	int cpu)
{
	cpu_data_t      *cdp = cpu_datap(cpu);
	boolean_t       intrs_enabled;
	uint64_t        tsc_timeout;

	/*
	 * Wait until the CPU indicates that it has stopped.
	 * Disable interrupts while the topo lock is held -- arguably
	 * this should always be done but in this instance it can lead to
	 * a timeout if long-running interrupt were to occur here.
	 */
	intrs_enabled = ml_set_interrupts_enabled(FALSE);
	mp_safe_spin_lock(&x86_topo_lock);
	/* Set a generous timeout of several seconds (in TSC ticks) */
	tsc_timeout = rdtsc64() + (10ULL * 1000 * 1000 * 1000);
	while ((cdp->lcpu.state != LCPU_HALT)
	    && (cdp->lcpu.state != LCPU_OFF)
	    && !cdp->lcpu.stopped) {
		simple_unlock(&x86_topo_lock);
		ml_set_interrupts_enabled(intrs_enabled);
		cpu_pause();
		if (rdtsc64() > tsc_timeout) {
			panic("cpu_exit_wait(%d) timeout", cpu);
		}
		ml_set_interrupts_enabled(FALSE);
		mp_safe_spin_lock(&x86_topo_lock);
	}
	simple_unlock(&x86_topo_lock);
	ml_set_interrupts_enabled(intrs_enabled);
}

void
cpu_machine_init(
	void)
{
	cpu_data_t      *cdp = current_cpu_datap();

	PE_cpu_machine_init(cdp->cpu_id, !cdp->cpu_boot_complete);
	cdp->cpu_boot_complete = TRUE;
	cdp->cpu_running = TRUE;
	ml_init_interrupt();

#if CONFIG_VMX
	/* initialize VMX for every CPU */
	vmx_cpu_init();
#endif
}

processor_t
current_processor(void)
{
	return current_cpu_datap()->cpu_processor;
}

processor_t
cpu_to_processor(
	int                     cpu)
{
	return cpu_datap(cpu)->cpu_processor;
}

ast_t *
ast_pending(void)
{
	return &current_cpu_datap()->cpu_pending_ast;
}

cpu_type_t
slot_type(
	int             slot_num)
{
	return cpu_datap(slot_num)->cpu_type;
}

cpu_subtype_t
slot_subtype(
	int             slot_num)
{
	return cpu_datap(slot_num)->cpu_subtype;
}

cpu_threadtype_t
slot_threadtype(
	int             slot_num)
{
	return cpu_datap(slot_num)->cpu_threadtype;
}

cpu_type_t
cpu_type(void)
{
	return current_cpu_datap()->cpu_type;
}

cpu_subtype_t
cpu_subtype(void)
{
	return current_cpu_datap()->cpu_subtype;
}

cpu_threadtype_t
cpu_threadtype(void)
{
	return current_cpu_datap()->cpu_threadtype;
}

const char *
processor_to_datastring(const char *prefix, processor_t target_processor)
{
	static char printBuf[256];
	uint32_t cpu_num = target_processor->cpu_id;

	cpu_data_t *cpup = cpu_datap(cpu_num);
	thread_t act;

	act = ml_validate_nofault((vm_offset_t)cpup->cpu_active_thread,
	    sizeof(struct thread)) ? cpup->cpu_active_thread : NULL;

	snprintf(printBuf, sizeof(printBuf),
	    "%s: tCPU %u (%d) [tid=0x%llx(bp=%d sp=%d) s=0x%x ps=0x%x cpa=0x%x spa=0x%llx pl=%d il=%d r=%d]",
	    prefix,
	    cpu_num,
	    target_processor->state,
	    act ? act->thread_id : ~0ULL,
	    act ? act->base_pri : -1,
	    act ? act->sched_pri : -1,
	    cpup->cpu_signals,
	    cpup->cpu_prior_signals,
	    cpup->cpu_pending_ast,
	    target_processor->processor_set->pending_AST_URGENT_cpu_mask,
	    cpup->cpu_preemption_level,
	    cpup->cpu_interrupt_level,
	    cpup->cpu_running);

	return (const char *)&printBuf[0];
}
