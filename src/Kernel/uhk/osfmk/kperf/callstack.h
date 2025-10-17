/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#ifndef KPERF_CALLSTACK_H
#define KPERF_CALLSTACK_H

#define MAX_KCALLSTACK_FRAMES (128)
#define MAX_UCALLSTACK_FRAMES (256)
#define MAX_EXCALLSTACK_FRAMES (128)

/* the callstack contains valid data */
#define CALLSTACK_VALID        (1U << 0)
/* the callstack has been deferred */
#define CALLSTACK_DEFERRED     (1U << 1)
/* the callstack is 64-bit */
#define CALLSTACK_64BIT        (1U << 2)
/* the callstack is from the kernel */
#define CALLSTACK_KERNEL       (1U << 3)
/* the callstack was cut off */
#define CALLSTACK_TRUNCATED    (1U << 4)
/* the callstack is only holding a continuation "frame" */
#define CALLSTACK_CONTINUATION (1U << 5)
/* the frames field is filled with uintptr_t, not uint64_t */
#define CALLSTACK_KERNEL_WORDS (1U << 6)
/* the frames come from a translated task */
#define CALLSTACK_TRANSLATED   (1U << 7)
/* the last frame could be the real PC */
#define CALLSTACK_FIXUP_PC     (1U << 8)
/* the stack also contains an async stack */
#define CALLSTACK_HAS_ASYNC    (1U << 9)

struct kp_ucallstack {
	uint32_t kpuc_flags;
	uint32_t kpuc_nframes;
	uint32_t kpuc_async_index;
	uint32_t kpuc_async_nframes;
	uintptr_t kpuc_frames[MAX_UCALLSTACK_FRAMES];
};

struct kp_kcallstack {
	uint32_t kpkc_flags;
	uint32_t kpkc_nframes;
	union {
		uintptr_t kpkc_word_frames[MAX_KCALLSTACK_FRAMES];
		uint64_t kpkc_frames[MAX_KCALLSTACK_FRAMES] __kernel_ptr_semantics;
	};
	uint32_t kpkc_exclaves_offset;
};

struct kperf_context;

void kperf_kcallstack_sample(struct kp_kcallstack *cs, struct kperf_context *);
void kperf_kcallstack_log(struct kp_kcallstack *cs);
void kperf_continuation_sample(struct kp_kcallstack *cs, struct kperf_context *);
void kperf_backtrace_sample(struct kp_kcallstack *cs, struct kperf_context *context);

void kperf_ucallstack_sample(struct kp_ucallstack *cs, struct kperf_context *);
int kperf_ucallstack_pend(struct kperf_context *, uint32_t depth,
    unsigned int actionid);
void kperf_ucallstack_log(struct kp_ucallstack *cs);

#if CONFIG_EXCLAVES
#include <kern/exclaves.tightbeam.h>
void kperf_excallstack_log(const stackshottypes_ipcstackentry_s *ipcstack);
bool kperf_exclave_callstack_pend(struct kperf_context *context, unsigned int actionid);
#endif /* CONFIG_EXCLAVES */

#endif /* !defined(KPERF_CALLSTACK_H) */
