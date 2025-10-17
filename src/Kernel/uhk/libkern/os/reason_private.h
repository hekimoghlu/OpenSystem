/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 21, 2024.
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
#ifndef OS_REASON_PRIVATE_H
#define OS_REASON_PRIVATE_H

#include <sys/reason.h>
#include <os/base.h>

__BEGIN_DECLS

#ifndef KERNEL

/*
 * similar to abort_with_payload, but for faults.
 *
 * [EBUSY]   too many corpses are being generated at the moment
 * [EQFULL]  the process used all its user fault quota
 * [ENOTSUP] generating simulated abort with reason is disabled
 * [EPERM]   generating simulated abort with reason for this namespace is not turned on
 */
int
os_fault_with_payload(uint32_t reason_namespace, uint64_t reason_code,
    void *payload, uint32_t payload_size, const char *reason_string,
    uint64_t reason_flags) __attribute__((cold));

#endif // !KERNEL

/*
 * Codes in the OS_REASON_LIBSYSTEM namespace
 */

OS_ENUM(os_reason_libsystem_code, uint64_t,
    OS_REASON_LIBSYSTEM_CODE_WORKLOOP_OWNERSHIP_LEAK = 1,
    OS_REASON_LIBSYSTEM_CODE_FAULT = 2, /* generic fault with old-style os_log_fault payload */
    OS_REASON_LIBSYSTEM_CODE_SECINIT_INITIALIZER = 3,
    OS_REASON_LIBSYSTEM_CODE_PTHREAD_CORRUPTION = 4,
    OS_REASON_LIBSYSTEM_CODE_OS_LOG_FAULT = 5, /* generated _only_ by os_log_fault in libtrace */
    );

__END_DECLS

#endif // OS_REASON_PRIVATE_H
