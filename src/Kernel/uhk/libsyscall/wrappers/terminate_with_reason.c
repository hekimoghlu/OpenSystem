/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 12, 2022.
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
#include <sys/reason.h>
#include <sys/types.h>
#include <stdint.h>
#include <signal.h>
#include <os/reason_private.h>
#include <unistd.h>

/* Crash simulation */

extern int pthread_current_stack_contains_np(const void *, unsigned long);
int
__darwin_check_fd_set_overflow(int n, const void *fd_set, int unlimited_select)
{
	if (n < 0) {
		os_fault_with_payload(OS_REASON_LIBSYSTEM, OS_REASON_LIBSYSTEM_CODE_FAULT,
		    &n, sizeof(n), "FD_SET underflow", 0);
		return 0;
	}

	if (n >= __DARWIN_FD_SETSIZE) {
		if (pthread_current_stack_contains_np((const void *) fd_set, sizeof(struct fd_set))) {
			if (!unlimited_select) {
				os_fault_with_payload(OS_REASON_LIBSYSTEM, OS_REASON_LIBSYSTEM_CODE_FAULT,
				    &n, sizeof(n), "FD_SET overflow", 0);
				return 0;
			} else {
				return 1;
			}
		} else {
			return 1;
		}
	}

	return 1;
}

/* System call entry points */
int __terminate_with_payload(int pid, uint32_t reason_namespace, uint64_t reason_code,
    void *payload, uint32_t payload_size, const char *reason_string,
    uint64_t reason_flags);

void __abort_with_payload(uint32_t reason_namespace, uint64_t reason_code,
    void *payload, uint32_t payload_size, const char *reason_string,
    uint64_t reason_flags);

static void abort_with_payload_wrapper_internal(uint32_t reason_namespace, uint64_t reason_code,
    void *payload, uint32_t payload_size, const char *reason_string,
    uint64_t reason_flags) __attribute__((noreturn, cold));

/* System call wrappers */
int
terminate_with_reason(int pid, uint32_t reason_namespace, uint64_t reason_code,
    const char *reason_string, uint64_t reason_flags)
{
	return __terminate_with_payload(pid, reason_namespace, reason_code, 0, 0,
	           reason_string, reason_flags);
}

int
terminate_with_payload(int pid, uint32_t reason_namespace, uint64_t reason_code,
    void *payload, uint32_t payload_size,
    const char *reason_string, uint64_t reason_flags)
{
	return __terminate_with_payload(pid, reason_namespace, reason_code, payload,
	           payload_size, reason_string, reason_flags);
}

static void
abort_with_payload_wrapper_internal(uint32_t reason_namespace, uint64_t reason_code,
    void *payload, uint32_t payload_size, const char *reason_string,
    uint64_t reason_flags)
{
	sigset_t unmask_signal;

	/* Try to unmask SIGABRT before trapping to the kernel */
	sigemptyset(&unmask_signal);
	sigaddset(&unmask_signal, SIGABRT);
	sigprocmask(SIG_UNBLOCK, &unmask_signal, NULL);

	__abort_with_payload(reason_namespace, reason_code, payload, payload_size,
	    reason_string, reason_flags);

	/* If sending a SIGABRT failed, we fall back to SIGKILL */
	terminate_with_payload(getpid(), reason_namespace, reason_code, payload, payload_size,
	    reason_string, reason_flags | OS_REASON_FLAG_ABORT);

	__builtin_unreachable();
}

void
abort_with_reason(uint32_t reason_namespace, uint64_t reason_code, const char *reason_string,
    uint64_t reason_flags)
{
	abort_with_payload_wrapper_internal(reason_namespace, reason_code, 0, 0, reason_string, reason_flags);
}

void
abort_with_payload(uint32_t reason_namespace, uint64_t reason_code, void *payload,
    uint32_t payload_size, const char *reason_string,
    uint64_t reason_flags)
{
	abort_with_payload_wrapper_internal(reason_namespace, reason_code, payload, payload_size,
	    reason_string, reason_flags);
}
