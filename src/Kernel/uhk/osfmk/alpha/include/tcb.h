/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 22, 2024.
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
 * Copyright (c) 2011 Philip Guenther <guenther@openbsd.org>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#ifndef _MACHINE_TCB_H_
#define _MACHINE_TCB_H_

#ifdef _KERNEL

void	*tcb_get(struct proc *_p);
void	tcb_set(struct proc *_p, void *_newtcb);

#define TCB_GET(p)		tcb_get(p)
#define TCB_SET(p, addr)	tcb_set(p, addr)

#else /* _KERNEL */

/* ELF TLS ABI calls for small TCB, with static TLS data after it */
#define TLS_VARIANT	1

static inline void *
__alpha_get_tcb(void)
{
	register void *__tmp __asm__("$0");

	__asm__ ("call_pal %1 # PAL_rdunique"
		: "=r" (__tmp)
		: "i" (0x009e /* PAL_rdunique */));
	return __tmp;
}
#define TCB_GET()		__alpha_get_tcb()

static inline void
__alpha_set_tcb(void *__val)
{
	register void *__tmp __asm__("$16") = __val;

	__asm__ volatile ("call_pal %1 # PAL_wrunique"
		: "=r" (__tmp)
		: "i" (0x009f /* PAL_wrunique */), "0" (__tmp));
}
#define TCB_SET(addr)		__alpha_set_tcb(addr)

#endif /* _KERNEL */

#endif /* _MACHINE_TCB_H_ */
