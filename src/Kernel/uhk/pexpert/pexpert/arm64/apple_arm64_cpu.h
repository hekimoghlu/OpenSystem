/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 13, 2024.
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
#pragma once

#if !defined(__ASSEMBLER__)

#include <stdbool.h>
#include <sys/cdefs.h>

/**
 * Defines the core type of the executing CPU.
 */
__enum_closed_decl(arm64_core_type_t, unsigned int, {
	E_CORE = MPIDR_CORETYPE_ACC_E,
	P_CORE = MPIDR_CORETYPE_ACC_P,
});

/*
 * Get the core type of the executing CPU.
 *
 * @return Whether the executing CPU is an E-core, P-core, or non-PE core.
 */
static inline arm64_core_type_t
arm64_core_type(void)
{
	return (arm64_core_type_t)((__builtin_arm_rsr64("MPIDR_EL1") >> MPIDR_CORETYPE_SHIFT) & MPIDR_CORETYPE_MASK);
}

/*
 * Convenience wrapper around arm64_core_type() which determines whether the
 * executing CPU is an E-core.
 *
 * @return Whether the executing CPU is an E-core.
 */
static inline bool
arm64_is_e_core(void)
{
	return arm64_core_type() == E_CORE;
}


/*
 * Convenience wrapper around arm64_core_type() which determines whether the
 * executing CPU is a P-core.
 *
 * @return Whether the executing CPU is a P-core.
 */
static inline bool
arm64_is_p_core(void)
{
	return arm64_core_type() == P_CORE;
}

/*
 * Convert a core type to a printable string.
 *
 * @param type The core type to convert.
 *
 * @return String describing whether the given core type corresponds to an
 *         E-core, P-core, or non-PE core.
 */
static inline const char *
arm64_core_type_to_string(arm64_core_type_t core_type)
{
	switch (core_type) {
	case E_CORE:
		return "E-core";
	case P_CORE:
		return "P-core";
	default:
		return "<< UNKNOWN OR INVALID CORE TYPE >>";
	}
}

/*
 * Convenience wrapper around arm64_core_type_to_string() which gets a printable
 * string describing the core type of the executing CPU.
 *
 * @return String describing whether the executing CPU is an E-core, P-core,
 *         or non-PE core.
 */
static inline const char *
arm64_core_type_as_string(void)
{
	return arm64_core_type_to_string(arm64_core_type());
}

#endif /* !defined(__ASSEMBLER__) */
