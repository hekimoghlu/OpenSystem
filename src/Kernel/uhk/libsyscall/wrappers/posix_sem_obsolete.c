/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 2, 2023.
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
#include <sys/errno.h>
#include <sys/semaphore.h>

/*
 * system call stubs are no longer generated for these from
 * syscalls.master. Instead, provide simple stubs here.
 */

int
sem_destroy(sem_t *s __unused)
{
	errno = ENOSYS;
	return -1;
}

int
sem_getvalue(sem_t * __restrict __unused s, int * __restrict __unused x)
{
	errno = ENOSYS;
	return -1;
}

int
sem_init(sem_t * __unused s, int __unused x, unsigned int __unused y)
{
	errno = ENOSYS;
	return -1;
}
