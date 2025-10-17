/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 1, 2025.
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
/**
 * On devices that support it, this test ensures that a mach exception is
 * generated when an ARMv8 floating point exception is triggered.
 * Also verifies that the main thread's FPCR value matches its expected default.
 */
#include <darwintest.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <mach/mach.h>
#include <mach/thread_status.h>
#include <sys/sysctl.h>
#include <inttypes.h>

#include "exc_helpers.h"

T_GLOBAL_META(
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("arm"),
	T_META_OWNER("devon_andrade"),
	T_META_RUN_CONCURRENTLY(true),
	T_META_TAG_VM_NOT_ELIGIBLE);

/* The bit to set in FPCR to enable the divide-by-zero floating point exception. */
#define FPCR_DIV_EXC 0x200
#define FPCR_INIT (0x0)

/* Whether we caught the EXC_ARITHMETIC mach exception or not. */
static volatile bool mach_exc_caught = false;

#ifdef __arm64__
static size_t
exc_arithmetic_handler(
	__unused mach_port_t task,
	__unused mach_port_t thread,
	exception_type_t type,
	mach_exception_data_t codes_64)
{
	/* Floating point divide by zero should cause an EXC_ARITHMETIC exception. */
	T_ASSERT_EQ(type, EXC_ARITHMETIC, "Caught an EXC_ARITHMETIC exception");

	/* Verify the exception is a floating point divide-by-zero exception. */
	T_ASSERT_EQ(codes_64[0], (mach_exception_data_type_t)EXC_ARM_FP_DZ, "The subcode is EXC_ARM_FP_DZ (floating point divide-by-zero)");

	mach_exc_caught = true;
	return 4;
}
#endif

#define KERNEL_BOOTARGS_MAX_SIZE 1024
static char kernel_bootargs[KERNEL_BOOTARGS_MAX_SIZE];

T_DECL(armv8_fp_exception,
    "Test that ARMv8 floating point exceptions generate Mach exceptions, verify default FPCR value.")
{
#ifndef __arm64__
	T_SKIP("Running on non-arm64 target, skipping...");
#else
	mach_port_t exc_port = MACH_PORT_NULL;
	size_t kernel_bootargs_len;

	uint64_t fpcr = __builtin_arm_rsr64("FPCR");

	if (fpcr != FPCR_INIT) {
		T_FAIL("The floating point control register has a non-default value" "%" PRIx64, fpcr);
	}

	/* Attempt to enable Divide-by-Zero floating point exceptions in hardware. */
	uint64_t fpcr_divexc = fpcr | FPCR_DIV_EXC;
	__builtin_arm_wsr64("FPCR", fpcr_divexc);
#define DSB_ISH 0xb
	__builtin_arm_dsb(DSB_ISH);

	/* Devices that don't support floating point exceptions have FPCR as RAZ/WI. */
	if (__builtin_arm_rsr64("FPCR") != fpcr_divexc) {
		T_SKIP("Running on a device that doesn't support floating point exceptions, skipping...");
	}

	/* Check if floating-point exceptions are enabled */
	kernel_bootargs_len = sizeof(kernel_bootargs);
	kern_return_t kr = sysctlbyname("kern.bootargs", kernel_bootargs, &kernel_bootargs_len, NULL, 0);
	if (kr != 0) {
		T_SKIP("Could not get kernel bootargs, skipping...");
	}

	if (NULL == strstr(kernel_bootargs, "-fp_exceptions")) {
		T_SKIP("Floating-point exceptions are disabled, skipping...");
	}

	/* Create the mach port the exception messages will be sent to. */
	exc_port = create_exception_port(EXC_MASK_ARITHMETIC);
	/* Spawn the exception server's thread. */
	run_exception_handler(exc_port, exc_arithmetic_handler);

	/**
	 * This should cause a floating point divide-by-zero exception to get triggered.
	 *
	 * The kernel shouldn't resume this thread until the mach exception is handled
	 * by the exception server that was just spawned. The exception handler will
	 * explicitly increment the PC += 4 to move to the next instruction.
	 */
	float a = 6.5f;
	float b = 0.0f;
	__asm volatile ("fdiv %s0, %s1, %s2" : "=w" (a) : "w" (a), "w" (b));

	if (mach_exc_caught) {
		T_PASS("The expected floating point divide-by-zero exception was caught!");
	} else {
		T_FAIL("The floating point divide-by-zero exception was not captured :(");
	}
#endif /* __arm64__ */
}
