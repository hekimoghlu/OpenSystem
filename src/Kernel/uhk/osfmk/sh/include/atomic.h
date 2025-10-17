/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 13, 2022.
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

#ifndef _SH_ATOMIC_H_
#define _SH_ATOMIC_H_

#if defined(_KERNEL)

#include <sh/psl.h>

static inline unsigned int
__atomic_enter(void)
{
	unsigned int sr;

	asm volatile ("stc sr, %0" : "=r"(sr));
	asm volatile ("ldc %0, sr" : : "r"(sr | PSL_IMASK));

	return (sr);
}

static inline void
__atomic_leave(unsigned int sr)
{
	asm volatile ("ldc %0, sr" : : "r"(sr));
}

static inline unsigned int
_atomic_cas_uint(volatile unsigned int *uip, unsigned int o, unsigned int n)
{
	unsigned int sr;
	unsigned int rv;

	sr = __atomic_enter();
	rv = *uip;
	if (rv == o)
		*uip = n;
	__atomic_leave(sr);

	return (rv);
}
#define atomic_cas_uint(_p, _o, _n) _atomic_cas_uint((_p), (_o), (_n))

static inline unsigned long
_atomic_cas_ulong(volatile unsigned long *uip, unsigned long o, unsigned long n)
{
	unsigned int sr;
	unsigned long rv;

	sr = __atomic_enter();
	rv = *uip;
	if (rv == o)
		*uip = n;
	__atomic_leave(sr);

	return (rv);
}
#define atomic_cas_ulong(_p, _o, _n) _atomic_cas_ulong((_p), (_o), (_n))

static inline void *
_atomic_cas_ptr(volatile void *uip, void *o, void *n)
{
	unsigned int sr;
	void * volatile *uipp = (void * volatile *)uip;
	void *rv;

	sr = __atomic_enter();
	rv = *uipp;
	if (rv == o)
		*uipp = n;
	__atomic_leave(sr);

	return (rv);
}
#define atomic_cas_ptr(_p, _o, _n) _atomic_cas_ptr((_p), (_o), (_n))

static inline unsigned int
_atomic_swap_uint(volatile unsigned int *uip, unsigned int n)
{
	unsigned int sr;
	unsigned int rv;

	sr = __atomic_enter();
	rv = *uip;
	*uip = n;
	__atomic_leave(sr);

	return (rv);
}
#define atomic_swap_uint(_p, _n) _atomic_swap_uint((_p), (_n))

static inline unsigned long
_atomic_swap_ulong(volatile unsigned long *uip, unsigned long n)
{
	unsigned int sr;
	unsigned long rv;

	sr = __atomic_enter();
	rv = *uip;
	*uip = n;
	__atomic_leave(sr);

	return (rv);
}
#define atomic_swap_ulong(_p, _n) _atomic_swap_ulong((_p), (_n))

static inline void *
_atomic_swap_ptr(volatile void *uip, void *n)
{
	unsigned int sr;
	void * volatile *uipp = (void * volatile *)uip;
	void *rv;

	sr = __atomic_enter();
	rv = *uipp;
	*uipp = n;
	__atomic_leave(sr);

	return (rv);
}
#define atomic_swap_ptr(_p, _n) _atomic_swap_ptr((_p), (_n))

static inline unsigned int
_atomic_add_int_nv(volatile unsigned int *uip, unsigned int v)
{
	unsigned int sr;
	unsigned int rv;

	sr = __atomic_enter();
	rv = *uip + v;
	*uip = rv;
	__atomic_leave(sr);

	return (rv);
}
#define atomic_add_int_nv(_p, _v) _atomic_add_int_nv((_p), (_v))

static inline unsigned long
_atomic_add_long_nv(volatile unsigned long *uip, unsigned long v)
{
	unsigned int sr;
	unsigned long rv;

	sr = __atomic_enter();
	rv = *uip + v;
	*uip = rv;
	__atomic_leave(sr);

	return (rv);
}
#define atomic_add_long_nv(_p, _v) _atomic_add_long_nv((_p), (_v))

static inline unsigned int
_atomic_sub_int_nv(volatile unsigned int *uip, unsigned int v)
{
	unsigned int sr;
	unsigned int rv;

	sr = __atomic_enter();
	rv = *uip - v;
	*uip = rv;
	__atomic_leave(sr);

	return (rv);
}
#define atomic_sub_int_nv(_p, _v) _atomic_sub_int_nv((_p), (_v))

static inline unsigned long
_atomic_sub_long_nv(volatile unsigned long *uip, unsigned long v)
{
	unsigned int sr;
	unsigned long rv;

	sr = __atomic_enter();
	rv = *uip - v;
	*uip = rv;
	__atomic_leave(sr);

	return (rv);
}
#define atomic_sub_long_nv(_p, _v) _atomic_sub_long_nv((_p), (_v))

static inline void
atomic_setbits_int(volatile unsigned int *uip, unsigned int v)
{
	unsigned int sr;

	sr = __atomic_enter();
	*uip |= v;
	__atomic_leave(sr);
}

static inline void
atomic_clearbits_int(volatile unsigned int *uip, unsigned int v)
{
	unsigned int sr;

	sr = __atomic_enter();
	*uip &= ~v;
	__atomic_leave(sr);
}

#endif /* defined(_KERNEL) */
#endif /* _SH_ATOMIC_H_ */
