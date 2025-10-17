/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 20, 2023.
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
/*
 *	File:	sys/aio_kern.h
 *	Author:	Jerry Cottingham [jerryc@apple.com]
 *
 *	Header file for kernel only portion of POSIX Asynchronous IO APIs
 *
 */

#include <sys/aio.h>

#ifndef _SYS_AIO_KERN_H_
#define _SYS_AIO_KERN_H_

#ifdef KERNEL_PRIVATE

typedef struct aio_workq_entry aio_workq_entry;

/*
 * Prototypes
 */

__private_extern__ void
_aio_close(struct proc *p, int fd);

__private_extern__ void
_aio_exit(struct proc *p);

__private_extern__ void
_aio_exec(struct proc *p);

__private_extern__ void
_aio_create_worker_threads(int num);

__private_extern__ void
aio_init(void);

task_t
get_aiotask(void);

#endif /* KERNEL_PRIVATE */

#endif /* _SYS_AIO_KERN_H_ */
