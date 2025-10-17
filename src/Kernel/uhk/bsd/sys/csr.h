/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 3, 2023.
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
#ifndef _SYS_CSR_H_
#define _SYS_CSR_H_

#include <stdint.h>
#include <sys/appleapiopts.h>
#include <sys/cdefs.h>

#ifdef __APPLE_API_PRIVATE

typedef uint32_t csr_config_t;
typedef uint32_t csr_op_t;

/* CSR configuration flags */
#define CSR_ALLOW_UNTRUSTED_KEXTS               (1 << 0)
#define CSR_ALLOW_UNRESTRICTED_FS               (1 << 1)
#define CSR_ALLOW_TASK_FOR_PID                  (1 << 2)
#define CSR_ALLOW_KERNEL_DEBUGGER               (1 << 3)
#define CSR_ALLOW_APPLE_INTERNAL                (1 << 4)
#define CSR_ALLOW_DESTRUCTIVE_DTRACE            (1 << 5) /* name deprecated */
#define CSR_ALLOW_UNRESTRICTED_DTRACE           (1 << 5)
#define CSR_ALLOW_UNRESTRICTED_NVRAM            (1 << 6)
#define CSR_ALLOW_DEVICE_CONFIGURATION          (1 << 7)
#define CSR_ALLOW_ANY_RECOVERY_OS               (1 << 8)
#define CSR_ALLOW_UNAPPROVED_KEXTS              (1 << 9)
#define CSR_ALLOW_EXECUTABLE_POLICY_OVERRIDE    (1 << 10)
#define CSR_ALLOW_UNAUTHENTICATED_ROOT          (1 << 11)
#define CSR_ALLOW_RESEARCH_GUESTS               (1 << 12)

#define CSR_VALID_FLAGS (CSR_ALLOW_UNTRUSTED_KEXTS | \
	                         CSR_ALLOW_UNRESTRICTED_FS | \
	                         CSR_ALLOW_TASK_FOR_PID | \
	                         CSR_ALLOW_KERNEL_DEBUGGER | \
	                         CSR_ALLOW_APPLE_INTERNAL | \
	                         CSR_ALLOW_UNRESTRICTED_DTRACE | \
	                         CSR_ALLOW_UNRESTRICTED_NVRAM | \
	                         CSR_ALLOW_DEVICE_CONFIGURATION | \
	                         CSR_ALLOW_ANY_RECOVERY_OS | \
	                         CSR_ALLOW_UNAPPROVED_KEXTS | \
	                         CSR_ALLOW_EXECUTABLE_POLICY_OVERRIDE | \
	                         CSR_ALLOW_UNAUTHENTICATED_ROOT | \
	                         CSR_ALLOW_RESEARCH_GUESTS)

#define CSR_ALWAYS_ENFORCED_FLAGS (CSR_ALLOW_DEVICE_CONFIGURATION | CSR_ALLOW_ANY_RECOVERY_OS)

/* Flags set by `csrutil disable`. */
#define CSR_DISABLE_FLAGS (CSR_ALLOW_UNTRUSTED_KEXTS | \
	                   CSR_ALLOW_UNRESTRICTED_FS | \
	                   CSR_ALLOW_TASK_FOR_PID | \
	                   CSR_ALLOW_KERNEL_DEBUGGER | \
	                   CSR_ALLOW_APPLE_INTERNAL | \
	                   CSR_ALLOW_UNRESTRICTED_DTRACE | \
	                   CSR_ALLOW_UNRESTRICTED_NVRAM)

/* CSR capabilities that a booter can give to the system */
#define CSR_CAPABILITY_UNLIMITED                        (1 << 0)
#define CSR_CAPABILITY_CONFIG                           (1 << 1)
#define CSR_CAPABILITY_APPLE_INTERNAL                   (1 << 2)

#define CSR_VALID_CAPABILITIES (CSR_CAPABILITY_UNLIMITED | CSR_CAPABILITY_CONFIG | CSR_CAPABILITY_APPLE_INTERNAL)

#ifdef PRIVATE
/* Private system call interface between Libsyscall and xnu */

/* Syscall flavors */
enum csr_syscalls {
	CSR_SYSCALL_CHECK,
	CSR_SYSCALL_GET_ACTIVE_CONFIG,
};

#endif /* PRIVATE */

__BEGIN_DECLS

/* Syscalls */
int csr_check(csr_config_t mask);
int csr_get_active_config(csr_config_t *config);

__END_DECLS

#endif /* __APPLE_API_PRIVATE */

#endif /* _SYS_CSR_H_ */
