/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 17, 2024.
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
#ifndef EXC_HELPERS_H
#define EXC_HELPERS_H

#include <mach/mach.h>
#include <mach/exception.h>
#include <stdbool.h>
#include <mach/thread_status.h>

/**
 * Callback invoked by run_exception_handler() when a Mach exception is
 * received.
 *
 * @param task      the task causing the exception
 * @param thread    the task causing the exception
 * @param type      exception type received from the kernel
 * @param codes     exception codes received from the kernel
 *
 * @return      how much the exception handler should advance the program
 *              counter, in bytes (in order to move past the code causing the
 *              exception)
 */
typedef size_t (*exc_handler_callback_t)(mach_port_t task, mach_port_t thread,
    exception_type_t type, mach_exception_data_t codes);

typedef size_t (*exc_handler_protected_callback_t)(task_id_token_t token, uint64_t thread_d,
    exception_type_t type, mach_exception_data_t codes);

typedef size_t (*exc_handler_state_protected_callback_t)(task_id_token_t token, uint64_t thread_d,
    exception_type_t type, mach_exception_data_t codes, thread_state_t in_state,
    mach_msg_type_number_t in_state_count, thread_state_t out_state, mach_msg_type_number_t *out_state_count);

typedef kern_return_t (*exc_handler_backtrace_callback_t)(kcdata_object_t kcdata_object,
    exception_type_t type, mach_exception_data_t codes);

/**
 * Allocates a Mach port and configures it to receive exception messages.
 *
 * @param exception_mask exception types that this Mach port should receive
 *
 * @return a newly-allocated and -configured Mach port
 */
mach_port_t
create_exception_port(exception_mask_t exception_mask);

mach_port_t
create_exception_port_behavior64(exception_mask_t exception_mask, exception_behavior_t behavior);

/**
 * Handles one exception received on the provided Mach port, by running the
 * provided callback.
 *
 * @param exc_port Mach port configured to receive exception messages
 * @param callback callback to run when an exception is received
 */
void
run_exception_handler(mach_port_t exc_port, exc_handler_callback_t callback);

void
run_exception_handler_behavior64(mach_port_t exc_port, void *preferred_callback, void *callback,
    exception_behavior_t behavior, bool run_once);

/**
 * Handles every exception received on the provided Mach port, by running the
 * provided callback.
 *
 * @param exc_port Mach port configured to receive exception messages
 * @param callback callback to run when an exception is received
 */
void
repeat_exception_handler(mach_port_t exc_port, exc_handler_callback_t callback);

#endif /* EXC_HELPERS_H */
