/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 9, 2024.
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
#ifndef _DTRACE_PTSS_H_
#define _DTRACE_PTSS_H_

#ifdef  __cplusplus
extern "C" {
#endif

/*
 * The pid provider needs a small per thread scratch space,
 * in the address space of the user task. This code is used to
 * manage that space.
 *
 * High level design:
 *
 * To avoid serialization, this is a (mostly) lockless allocator. If
 * a new page has to be allocated, the process's sprlock will be acquired.
 *
 * NOTE: The dtrace copyin/copyout code is still the shared code that
 * can handle unmapped pages, so the scratch space isn't wired for now.
 * * Each page in user space is wired. It cannot be paged out, because
 * * dtrace's copyin/copyout is only guaranteed to handle pages already
 * * in memory.
 *
 * Each page in user space is represented by a dt_ptss_page. Page entries
 * are chained. Once allocated, a page is not freed until dtrace "cleans up"
 * that process.
 *
 * Clean up works like this:
 *
 * At process exit, free all kernel allocated memory, but ignore user pages.
 * At process exec, free all kernel allocated memory, but ignore user pages.
 * At process fork, free user pages copied from parent, and do not allocate kernel memory.
 *
 * This is making the assumption that it is faster to let the bulk vm_map
 * operations in exec/exit do their work, instead of explicit page free(s)
 * via mach_vm_deallocate.
 *
 * As each page is allocated, its entries are chained and added to the
 * free_list. To claim an entry, cas it off the list. When a thread exits,
 * cas its entry onto the list. We could potentially optimize this by
 * keeping a head/tail, and cas'ing the frees to the tail instead of the
 * head. Without evidence to support such a need, it seems better to keep
 * things simple for now.
 */


#define DTRACE_PTSS_SCRATCH_SPACE_PER_THREAD (64)

#define DTRACE_PTSS_ENTRIES_PER_PAGE (PAGE_MAX_SIZE / DTRACE_PTSS_SCRATCH_SPACE_PER_THREAD)

struct dtrace_ptss_page_entry {
	struct dtrace_ptss_page_entry*  next;
	user_addr_t                     addr;
	user_addr_t                     write_addr;
};

struct dtrace_ptss_page {
	struct dtrace_ptss_page*       next;
	struct dtrace_ptss_page_entry  entries[DTRACE_PTSS_ENTRIES_PER_PAGE];
};

struct dtrace_ptss_page_entry*  dtrace_ptss_claim_entry(struct proc* p); /* sprlock not held */
struct dtrace_ptss_page_entry*  dtrace_ptss_claim_entry_locked(struct proc* p); /* sprlock held */
void                            dtrace_ptss_release_entry(struct proc* p, struct dtrace_ptss_page_entry* e);

struct dtrace_ptss_page*        dtrace_ptss_allocate_page(struct proc* p);
void                            dtrace_ptss_free_page(struct proc* p, struct dtrace_ptss_page* ptss_page);

void                            dtrace_ptss_enable(struct proc* p);
void                            dtrace_ptss_exec_exit(struct proc* p);
void                            dtrace_ptss_fork(struct proc* parent, struct proc* child);

#ifdef  __cplusplus
}
#endif

#endif  /* _DTRACE_PTSS_H_ */
