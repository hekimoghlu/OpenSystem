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
#include <mach/vm_types.h>
#include <i386/acpi.h> /* install_real_mode_bootstrap */
#include <i386/mp.h>
#include <i386/lapic.h> /* lapic_* functions */
#include <i386/machine_routines.h>
#include <i386/cpu_data.h>
#include <i386/pmap.h>
#include <i386/bit_routines.h>

/* PAL-related routines */
void i386_cpu_IPI(int cpu);
boolean_t i386_smp_init(int nmi_vector, i386_intr_func_t nmi_handler,
    int ipi_vector, i386_intr_func_t ipi_handler);
void i386_start_cpu(int lapic_id, int cpu_num);
void i386_send_NMI(int cpu);
void handle_pending_TLB_flushes(void);
void NMIPI_enable(boolean_t);

extern void     slave_pstart(void);

#ifdef  MP_DEBUG
int     trappedalready = 0;     /* (BRINGUP) */
#endif  /* MP_DEBUG */

boolean_t
i386_smp_init(int nmi_vector, i386_intr_func_t nmi_handler, int ipi_vector, i386_intr_func_t ipi_handler)
{
	/* Local APIC? */
	if (!lapic_probe()) {
		return FALSE;
	}

	lapic_init();
	lapic_configure(false);
	lapic_set_intr_func(nmi_vector, nmi_handler);
	lapic_set_intr_func(ipi_vector, ipi_handler);

	install_real_mode_bootstrap(slave_pstart);

	return TRUE;
}

void
i386_start_cpu(int lapic_id, __unused int cpu_num )
{
	LAPIC_WRITE_ICR(lapic_id, LAPIC_ICR_DM_INIT);
	delay(100);
	LAPIC_WRITE_ICR(lapic_id,
	    LAPIC_ICR_DM_STARTUP | (REAL_MODE_BOOTSTRAP_OFFSET >> 12));
}

static boolean_t NMIPIs_enabled = FALSE;

void
NMIPI_enable(boolean_t enable)
{
	NMIPIs_enabled = enable;
}

void
i386_send_NMI(int cpu)
{
	boolean_t state = ml_set_interrupts_enabled(FALSE);

	if (NMIPIs_enabled == FALSE) {
		i386_cpu_IPI(cpu);
	} else {
		lapic_send_nmi(cpu);
	}
	(void) ml_set_interrupts_enabled(state);
}

void
handle_pending_TLB_flushes(void)
{
	volatile int    *my_word = &current_cpu_datap()->cpu_signals;

	if (i_bit(MP_TLB_FLUSH, my_word) && (pmap_tlb_flush_timeout == FALSE)) {
		DBGLOG(cpu_handle, cpu_number(), MP_TLB_FLUSH);
		i_bit_clear(MP_TLB_FLUSH, my_word);
		pmap_update_interrupt();
	}
}

void
i386_cpu_IPI(int cpu)
{
#ifdef  MP_DEBUG
	if (cpu_datap(cpu)->cpu_signals & 6) {   /* (BRINGUP) */
		kprintf("i386_cpu_IPI: sending enter debugger signal (%08X) to cpu %d\n", cpu_datap(cpu)->cpu_signals, cpu);
	}
#endif  /* MP_DEBUG */

	lapic_send_ipi(cpu, LAPIC_VECTOR(INTERPROCESSOR));
}
