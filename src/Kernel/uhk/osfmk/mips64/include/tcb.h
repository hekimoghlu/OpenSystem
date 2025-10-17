/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 6, 2025.
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

static inline void
__mips64_set_tcb(struct proc *p, void *tcb)
{
#ifdef CPU_MIPS64R2
	extern int cpu_has_userlocal;

	if (cpu_has_userlocal)
		cp0_set_userlocal(tcb);
#endif

	p->p_md.md_tcb = tcb;
}

#define TCB_SET(p, addr)	__mips64_set_tcb(p, addr)
#define TCB_GET(p)		((p)->p_md.md_tcb)

#else /* _KERNEL */

/* ELF TLS ABI calls for small TCB, with static TLS data after it */
#define TLS_VARIANT	1

static inline void *
__mips64_get_tcb(void)
{
	void *tcb;

	/*
	 * This invokes emulation in kernel if the system does not implement
	 * the RDHWR instruction or the UserLocal register.
	 */
	__asm__ volatile (
	"	.set	push\n"
	"	.set	mips64r2\n"
	"	rdhwr	%0, $29\n"
	"	.set	pop\n" : "=r" (tcb));
	return tcb;
}

#define TCB_GET()		__mips64_get_tcb()

#endif /* _KERNEL */

#endif /* _MACHINE_TCB_H_ */
