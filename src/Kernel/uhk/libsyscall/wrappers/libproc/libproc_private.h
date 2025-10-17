/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 9, 2023.
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
#ifndef _LIBPROC_PRIVATE_H_
#define _LIBPROC_PRIVATE_H_

#include <libproc.h>

#if defined(PRIVATE) && \
        defined(_LIBPROC_PRIVATE_H_) /* Defeat unifdef */
#include <sys/event_private.h>

__BEGIN_DECLS

/*
 * Enumerate potential userspace pointers embedded in kernel data structures.
 * Currently inspects kqueues only.
 *
 * NOTE: returned "pointers" are opaque user-supplied values and thus not
 * guaranteed to address valid objects or be pointers at all.
 *
 * Returns the number of pointers found (which may exceed buffersize), or -1 on
 * failure and errno set appropriately.
 */
int proc_list_uptrs(pid_t pid, uint64_t *buffer, uint32_t buffersize);

int proc_list_dynkqueueids(int pid, kqueue_id_t *buf, uint32_t bufsz);
int proc_piddynkqueueinfo(int pid, int flavor, kqueue_id_t kq_id, void *buffer,
    int buffersize);

__END_DECLS

#endif /* PRIVATE */

#endif /* _LIBPROC_PRIVATE_H_ */
