/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 14, 2024.
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
 * UBSan minimal runtime for production environments
 * This runtime emulates an inlined trap model, but gated through the
 * logging functions, so that fix and continue is always possible just by
 * advancing the program counter.
 * NOTE: this file is regularly instrumented, since all helpers must inline
 * correctly to a trap.
 */
#include <sys/sysctl.h>
#include <kern/kalloc.h>
#include <kern/trap_telemetry.h>
#include <machine/machine_routines.h>
#include <san/ubsan_minimal.h>

#define UBSAN_M_NONE            (0x0000)
#define UBSAN_M_PANIC           (0x0001)

#define UBSAN_MINIMAL_TRAPS_START       UBSAN_SOFT_TRAP_SIGNED_OF
#define UBSAN_MINIMAL_TRAPS_END         UBSAN_SOFT_TRAP_SIGNED_OF

struct ubsan_minimal_trap_desc {
	uint16_t        id;
	uint32_t        flags;
	char            str[16];
};

#if RELEASE
static __security_const_late
#endif /* RELEASE */
struct ubsan_minimal_trap_desc ubsan_traps[] = {
	{ UBSAN_SOFT_TRAP_SIGNED_OF, UBSAN_M_NONE, "signed-overflow" },
};

static SECURITY_READ_ONLY_LATE(bool) ubsan_minimal_enabled = false;

/* Helper counters for fixup/drop operations */
static uint32_t ubsan_minimal_fixup_events;

static char *
ubsan_minimal_trap_to_str(uint16_t trap)
{
	return ubsan_traps[trap - UBSAN_MINIMAL_TRAPS_START].str;
}

void
ubsan_handle_brk_trap(
	__unused void     *state,
	uint16_t          trap)
{
	if (!ubsan_minimal_enabled) {
#if DEVELOPMENT
		/* We want to know about those sooner than later... */
		panic("UBSAN trap taken in early startup code");
#endif /* DEVELOPMENT */
		return;
	}

	uint32_t trap_idx = trap - UBSAN_MINIMAL_TRAPS_START;

	if (ubsan_traps[trap_idx].flags & UBSAN_M_PANIC) {
		panic("UBSAN trap for %s detected\n", ubsan_minimal_trap_to_str(trap));
		__builtin_unreachable();
	}

	ubsan_minimal_fixup_events++;
}

__startup_func
void
ubsan_minimal_init(void)
{
#if DEVELOPMENT || DEBUG
	bool force_panic = false;
	PE_parse_boot_argn("-ubsan_force_panic", &force_panic, sizeof(force_panic));
	if (force_panic) {
		for (int i = UBSAN_MINIMAL_TRAPS_START; i <= UBSAN_MINIMAL_TRAPS_END; i++) {
			size_t idx = i - UBSAN_MINIMAL_TRAPS_START;
			ubsan_traps[idx].flags |= UBSAN_M_PANIC;
		}
	}
#endif /* DEVELOPMENT || DEBUG */

	ubsan_minimal_enabled = true;
}

#if DEVELOPMENT || DEBUG

/* Add a simple testing path that explicitly triggers a signed int overflow */
static int
sysctl_ubsan_test SYSCTL_HANDLER_ARGS
{
#pragma unused(oidp, arg1, arg2)
	int err, incr = 0;
	err = sysctl_io_number(req, 0, sizeof(int), &incr, NULL);

	int k = 0x7fffffff;

	for (int i = 0; i < 3; i++) {
		int a = k;
		if (incr != 0) {
			k += incr;
		}

		k = a;
	}

	return err;
}

SYSCTL_DECL(kern_ubsan);
SYSCTL_NODE(_kern, OID_AUTO, ubsan, CTLFLAG_RW | CTLFLAG_LOCKED, 0, "ubsan");
SYSCTL_NODE(_kern_ubsan, OID_AUTO, minimal, CTLFLAG_RW | CTLFLAG_LOCKED, 0, "minimal runtime");
SYSCTL_INT(_kern_ubsan_minimal, OID_AUTO, fixups, CTLFLAG_RW | CTLFLAG_LOCKED, &ubsan_minimal_fixup_events, 0, "");
SYSCTL_INT(_kern_ubsan_minimal, OID_AUTO, signed_ovf_flags, CTLFLAG_RW | CTLFLAG_LOCKED, &(ubsan_traps[0].flags), 0, "");
SYSCTL_PROC(_kern_ubsan_minimal, OID_AUTO, test, CTLTYPE_INT | CTLFLAG_RW | CTLFLAG_LOCKED,
    0, 0, sysctl_ubsan_test, "I", "Test signed overflow detection");

#endif /* DEVELOPMENT || DEBUG */

#define UBSAN_M_ATTR    __attribute__((always_inline, cold))

UBSAN_M_ATTR void
__ubsan_handle_divrem_overflow_minimal(void)
{
	asm volatile ("brk #%0" : : "i"(UBSAN_SOFT_TRAP_SIGNED_OF));
}

UBSAN_M_ATTR void
__ubsan_handle_negate_overflow_minimal(void)
{
	asm volatile ("brk #%0" : : "i"(UBSAN_SOFT_TRAP_SIGNED_OF));
}

UBSAN_M_ATTR void
__ubsan_handle_mul_overflow_minimal(void)
{
	asm volatile ("brk #%0" : : "i"(UBSAN_SOFT_TRAP_SIGNED_OF));
}

UBSAN_M_ATTR void
__ubsan_handle_sub_overflow_minimal(void)
{
	asm volatile ("brk #%0" : : "i"(UBSAN_SOFT_TRAP_SIGNED_OF));
}

UBSAN_M_ATTR void
__ubsan_handle_add_overflow_minimal(void)
{
	asm volatile ("brk #%0" : : "i"(UBSAN_SOFT_TRAP_SIGNED_OF));
}
