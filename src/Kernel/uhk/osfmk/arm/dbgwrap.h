/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 16, 2023.
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

#include <mach/thread_status.h>
#include <sys/cdefs.h>

__BEGIN_DECLS

#if defined(__arm64__)
typedef arm_thread_state64_t __attribute__((aligned(16))) dbgwrap_thread_state_t;
#else
typedef arm_thread_state32_t dbgwrap_thread_state_t;
#endif

typedef enum {
	DBGWRAP_ERR_SELF_HALT = -6,
	DBGWRAP_ERR_UNSUPPORTED = -5,
	DBGWRAP_ERR_INPROGRESS = -4,
	DBGWRAP_ERR_INSTR_ERROR = -3,
	DBGWRAP_ERR_INSTR_TIMEOUT = -2,
	DBGWRAP_ERR_HALT_TIMEOUT = -1,
	DBGWRAP_SUCCESS = 0,
	DBGWRAP_WARN_ALREADY_HALTED,
	DBGWRAP_WARN_CPU_OFFLINE
} dbgwrap_status_t;

static inline const char*
ml_dbgwrap_strerror(dbgwrap_status_t status)
{
	switch (status) {
	case DBGWRAP_ERR_SELF_HALT:             return "CPU attempted to halt itself";
	case DBGWRAP_ERR_UNSUPPORTED:           return "halt not supported for this configuration";
	case DBGWRAP_ERR_INPROGRESS:            return "halt in progress on another CPU";
	case DBGWRAP_ERR_INSTR_ERROR:           return "instruction-stuffing failure";
	case DBGWRAP_ERR_INSTR_TIMEOUT:         return "instruction-stuffing timeout";
	case DBGWRAP_ERR_HALT_TIMEOUT:          return "halt ack timeout, CPU likely wedged";
	case DBGWRAP_SUCCESS:                   return "halt succeeded";
	case DBGWRAP_WARN_ALREADY_HALTED:       return "CPU already halted";
	case DBGWRAP_WARN_CPU_OFFLINE:          return "CPU offline";
	default:                                return "unrecognized status";
	}
}

boolean_t ml_dbgwrap_cpu_is_halted(int cpu_index);

dbgwrap_status_t ml_dbgwrap_wait_cpu_halted(int cpu_index, uint64_t timeout_ns);

dbgwrap_status_t ml_dbgwrap_halt_cpu(int cpu_index, uint64_t timeout_ns);

dbgwrap_status_t ml_dbgwrap_halt_cpu_with_state(int cpu_index, uint64_t timeout_ns, dbgwrap_thread_state_t *state);

__END_DECLS
