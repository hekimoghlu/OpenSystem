/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 4, 2022.
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
/* Public Domain */

#ifndef _MACHINE_ATOMIC_H_
#define _MACHINE_ATOMIC_H_

#define __membar(_f) do {__asm volatile(_f ::: "memory"); } while (0)

#define membar_enter()		__membar("fence w,rw")
#define membar_exit()		__membar("fence rw,w")
#define membar_producer()	__membar("fence w,w")
#define membar_consumer()	__membar("fence r,r")
#define membar_sync()		__membar("fence rw,rw")

#if defined(_KERNEL)

/* virtio needs MP membars even on SP kernels */
#define virtio_membar_producer()	__membar("fence w,w")
#define virtio_membar_consumer()	__membar("fence r,r")
#define virtio_membar_sync()		__membar("fence rw,rw")

/*
 * Set bits
 * *p = *p | v
 */
static inline void
atomic_setbits_int(volatile unsigned int *p, unsigned int v)
{
	__asm volatile("amoor.w zero, %1, %0"
			: "+A" (*p)
			: "r" (v)
			: "memory");
}

/*
 * Clear bits
 * *p = *p & (~v)
 */
static inline void
atomic_clearbits_int(volatile unsigned int *p, unsigned int v)
{
	__asm volatile("amoand.w zero, %1, %0"
			: "+A" (*p)
			: "r" (~v)
			: "memory");
}

#endif /* defined(_KERNEL) */
#endif /* _MACHINE_ATOMIC_H_ */
