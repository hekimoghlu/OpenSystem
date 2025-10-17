/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 19, 2022.
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
#include <sys/cdefs.h>
#include <stdint.h>

__BEGIN_DECLS

#ifdef KERNEL
__abortlike __printflike(1, 2)
extern void panic(const char *string, ...);
#endif /* KERNEL */

#if KERNEL_PRIVATE
struct task;
struct thread;
struct proc;

#if XNU_KERNEL_PRIVATE
#define panic(ex, ...)  ({ \
	__asm__("" ::: "memory"); \
	(panic)(ex " @%s:%d", ## __VA_ARGS__, __FILE_NAME__, __LINE__); \
})
#else /* else XNU_KERNEL_PRIVATE */
#define panic(ex, ...)  ({ \
	__asm__("" ::: "memory"); \
	(panic)(#ex " @%s:%d", ## __VA_ARGS__, __FILE_NAME__, __LINE__); \
})
#endif /* else XNU_KERNEL_PRIVATE*/
#define panic_plain(ex, ...)  (panic)(ex, ## __VA_ARGS__)

__abortlike __printflike(4, 5)
void panic_with_options(unsigned int reason, void *ctx,
    uint64_t debugger_options_mask, const char *str, ...);
__abortlike __printflike(5, 6)
void panic_with_options_and_initiator(const char* initiator, unsigned int reason, void *ctx,
    uint64_t debugger_options_mask, const char *str, ...);

#if XNU_KERNEL_PRIVATE && defined (__x86_64__)
__abortlike __printflike(5, 6)
void panic_with_thread_context(unsigned int reason, void *ctx,
    uint64_t debugger_options_mask, struct thread* th, const char *str, ...);
#endif /* XNU_KERNEL_PRIVATE && defined (__x86_64__) */

#endif /* KERNEL_PRIVATE */

__END_DECLS
