/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 4, 2024.
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
#include "timingsafe.h"
#include <machine/cpu_capabilities.h>
#include <os/base.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/cdefs.h>
#include <sys/sysctl.h>
#include <unistd.h>

#define REG_DIT "dit"
#define REG_SB "sb"
#define DIT_ON (1)
#define DIT_OFF (0)
#define DIT_REG_MASK (1U << 24)
#define BARRIER_SY (0xf)
#define COMMPAGE_FEATURES                                                      \
	((volatile uint64_t const *const)_COMM_PAGE_CPU_CAPABILITIES64)
#define TEST_FEATURES(FEATURES) ((*(COMMPAGE_FEATURES)) & (FEATURES))

// Use commpage for features that aren't mandatory for the current
// architecture.
#if defined(__ARM_ARCH_8_5__)
#define NEED_FEATURES (0)
#define HAS_FEATURES (kHasFeatDIT | kHasFeatSB)
#elif defined(__ARM_ARCH_8_4__)
#define NEED_FEATURES (kHasFeatSB)
#define HAS_FEATURES (kHasFeatDIT)
#else
#define HAS_FEATURES (0)
#define NEED_FEATURES (kHasFeatSB | kHasFeatDIT)
#endif

#if NEED_FEATURES
#define CPU_FEATURES ((HAS_FEATURES) | TEST_FEATURES(NEED_FEATURES))
#else
#define CPU_FEATURES (HAS_FEATURES)
#endif

/**
 Definition of supported CPU features.

 timingsafe_token_t is an opaque wrapper around this type, so this type must remain compatible.
 */
OS_OPTIONS(timingsafe_features, uint64_t,
	/**
	 No timingsafe features supported.
	 */
	TIMINGSAFE_FEATURE_NONE = 0,

	/**
	 Data Independent Timing feature.
	 More information on this feature is available at the following link:
	 https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms#Enable-DIT-for-constant-time-cryptographic-operations
	 */
	TIMINGSAFE_FEATURE_DIT = 1,
);

/**
 CPU capabilities from commpage.
 */
typedef uint64_t cpu_cap_t;

__attribute__((target(REG_DIT))) static inline bool is_dit_enabled(void) {
	return 0 != (DIT_REG_MASK & __builtin_arm_rsr64(REG_DIT));
}

__attribute__((target(REG_SB))) static inline void sb(void) {
	asm volatile(REG_SB::: "memory");
}

static inline void speculation_barrier(cpu_cap_t feat) {
	if (feat & kHasFeatSB) {
		sb();
	} else {
		__builtin_arm_dsb(BARRIER_SY);
		__builtin_arm_isb(BARRIER_SY);
	}
}

__attribute__((target(REG_DIT))) timingsafe_token_t
timingsafe_enable_if_supported(void) {
	timingsafe_features_t token = TIMINGSAFE_FEATURE_NONE;
	cpu_cap_t const feat = CPU_FEATURES;
	if (feat & kHasFeatDIT) {
		if (is_dit_enabled()) {
			token |= TIMINGSAFE_FEATURE_DIT;
		}
		__builtin_arm_wsr64(REG_DIT, DIT_ON);
	}
	speculation_barrier(feat);
	return (timingsafe_token_t)token;
}

__attribute__((target(REG_DIT))) void
timingsafe_restore_if_supported(timingsafe_token_t token) {
	timingsafe_features_t const enum_token = (timingsafe_features_t)token;
	cpu_cap_t const feat = CPU_FEATURES;
	if ((feat & kHasFeatDIT) && !(enum_token & TIMINGSAFE_FEATURE_DIT)) {
		// Disable DIT if it was previously disabled
		__builtin_arm_wsr64(REG_DIT, DIT_OFF);
	}
}
