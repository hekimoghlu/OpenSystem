/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 17, 2025.
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
#ifndef _SYS_PTHREAD_INTERNAL_H_
#define _SYS_PTHREAD_INTERNAL_H_

#include <sys/user.h>
#include <kern/thread_call.h>

struct ksyn_waitq_element {
#if __LP64__
	char opaque[48];
#else
	char opaque[32];
#endif
};

void workq_mark_exiting(struct proc *);
void workq_exit(struct proc *);
void pthread_init(void);

#endif /* _SYS_PTHREAD_INTERNAL_H_ */
