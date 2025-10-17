/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 12, 2021.
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

#include <mach/exclaves.h>
#include <mach/kern_return.h>
#include <kern/kern_types.h>
#include <kern/kern_cdata.h>
#include <kern/thread.h>
#include <sys/cdefs.h>

#if CONFIG_EXCLAVES

#include <kern/exclaves.tightbeam.h>

__BEGIN_DECLS

/*
 * Kick the collection thread to ensure it's running.
 */
extern void exclaves_inspection_begin_collecting(void);
/*
 * Wait for provided queue to drain.
 */
extern void exclaves_inspection_wait_complete(queue_t exclaves_inspection_queue);

extern void exclaves_inspection_check_ast(void);

extern bool exclaves_stackshot_raw_addresses;
extern bool exclaves_stackshot_all_address_spaces;

extern lck_mtx_t exclaves_collect_mtx;
/*
 * These waitlists are protected by exclaves_collect_mtx and should not be
 * cleared other than by the dedicated `exclaves_collection_thread` thread.
 */
extern queue_head_t exclaves_inspection_queue_stackshot;
extern queue_head_t exclaves_inspection_queue_kperf;

static inline void
exclaves_inspection_queue_add(queue_t queue, queue_entry_t elm)
{
	assert(queue == &exclaves_inspection_queue_stackshot || queue == &exclaves_inspection_queue_kperf);
	lck_mtx_assert(&exclaves_collect_mtx, LCK_ASSERT_OWNED);

	enqueue_head(queue, elm);
}

struct exclaves_panic_stackshot {
	uint8_t *stackshot_buffer;
	uint64_t stackshot_buffer_size;
};

__enum_decl(exclaves_panic_ss_status_t, uint8_t, {
	EXCLAVES_PANIC_STACKSHOT_UNKNOWN = 0,
	EXCLAVES_PANIC_STACKSHOT_FOUND = 1,
	EXCLAVES_PANIC_STACKSHOT_NOT_FOUND = 2,
	EXCLAVES_PANIC_STACKSHOT_DECODE_FAILED = 3,
});

extern exclaves_panic_ss_status_t exclaves_panic_ss_status;

/* Attempt to read Exclave panic stackshot data */
void kdp_read_panic_exclaves_stackshot(struct exclaves_panic_stackshot *eps);

/* Convert exclaves stackshot data from tightbeam structures into kcdata. */
kern_return_t
stackshot_exclaves_process_stackshot(const stackshot_stackshotresult_s *result, void *kcdata_ptr, bool want_raw_addresses);

__END_DECLS

#endif /* CONFIG_EXCLAVES */
