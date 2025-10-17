/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 29, 2024.
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
#include <stdlib.h>
#include <darwintest.h>
#include <mach/mach.h>
#include <mach/thread_status.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.arm"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("arm"),
	T_META_OWNER("justin_unger"),
	T_META_RUN_CONCURRENTLY(true)
	);

#define PSR64_USER_MASK (0xFU << 28)
#define PSR64_OPT_BITS  (0x01 << 12) // user-writeable bits that may or may not be set, depending on hardware/device/OS/moon phase

#if __arm64__
__attribute__((noreturn))
static void
phase2()
{
	kern_return_t err;
	arm_thread_state64_t ts;
	mach_msg_type_number_t count = ARM_THREAD_STATE64_COUNT;
	uint32_t nzcv = (uint32_t) __builtin_arm_rsr64("NZCV");

	T_QUIET; T_ASSERT_EQ(nzcv & PSR64_USER_MASK, PSR64_USER_MASK, "All condition flags are set");

	err = thread_get_state(mach_thread_self(), ARM_THREAD_STATE64, (thread_state_t)&ts, &count);
	T_QUIET; T_ASSERT_EQ(err, KERN_SUCCESS, "Got own thread state after corrupting CPSR");

	T_QUIET; T_ASSERT_EQ(ts.__cpsr & ~(PSR64_USER_MASK | PSR64_OPT_BITS), 0, "No privileged fields in CPSR are set");

	exit(0);
}
#endif

T_DECL(thread_set_state_arm64_cpsr,
    "Test that user mode cannot control privileged fields in CPSR/PSTATE.", T_META_TAG_VM_NOT_ELIGIBLE)
{
#if !__arm64__
	T_SKIP("Running on non-arm64 target, skipping...");
#else
	kern_return_t err;
	mach_msg_type_number_t count;
	arm_thread_state64_t ts;

	count = ARM_THREAD_STATE64_COUNT;
	err = thread_get_state(mach_thread_self(), ARM_THREAD_STATE64, (thread_state_t)&ts, &count);
	T_QUIET; T_ASSERT_EQ(err, KERN_SUCCESS, "Got own thread state");

	/*
	 * jump to the second phase while attempting to set all the bits
	 * in CPSR. If we survive the jump and read back CPSR without any
	 * bits besides condition flags set, the test passes. If kernel
	 * does not mask out the privileged CPSR bits correctly, we can
	 * expect an illegal instruction set panic due to SPSR.IL being
	 * set upon ERET to user mode.
	 */

	void *new_pc = (void *)&phase2;
	arm_thread_state64_set_pc_fptr(ts, new_pc);
	ts.__cpsr = ~0U;

	err = thread_set_state(mach_thread_self(), ARM_THREAD_STATE64, (thread_state_t)&ts, ARM_THREAD_STATE64_COUNT);

	/* NOT REACHED */

	T_ASSERT_FAIL("Thread did not reach expected state. err = %d", err);

#endif
}
