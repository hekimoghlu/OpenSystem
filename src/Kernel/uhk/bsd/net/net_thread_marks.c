/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 20, 2025.
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
#include "kpi_interface.h"
#include <stddef.h>

#include <net/dlil.h>

#include <sys/param.h>
#include <sys/systm.h>
#include <sys/kernel.h>
#include <sys/malloc.h>
#include <sys/mbuf.h>
#include <sys/socket.h>
#include <sys/domain.h>
#include <sys/user.h>

#include <kern/assert.h>
#include <kern/task.h>
#include <kern/thread.h>
#include <kern/sched_prim.h>
#include <kern/locks.h>
#include <kern/zalloc.h>

struct net_thread_marks { };
static const struct net_thread_marks net_thread_marks_base = { };

__private_extern__ const net_thread_marks_t net_thread_marks_none =
    &net_thread_marks_base;

__private_extern__ net_thread_marks_t
net_thread_marks_push(u_int32_t push)
{
	static const char *__unsafe_indexable const base = (const void*__single)&net_thread_marks_base;
	u_int32_t pop = 0;

	if (push != 0) {
		struct uthread *uth = current_uthread();

		pop = push & ~uth->uu_network_marks;
		if (pop != 0) {
			uth->uu_network_marks |= pop;
		}
	}

	return __unsafe_forge_single(net_thread_marks_t, (base + pop));
}

__private_extern__ net_thread_marks_t
net_thread_unmarks_push(u_int32_t unpush)
{
	static const char *__unsafe_indexable const base = (const void*__single)&net_thread_marks_base;
	u_int32_t unpop = 0;

	if (unpush != 0) {
		struct uthread *uth = current_uthread();

		unpop = unpush & uth->uu_network_marks;
		if (unpop != 0) {
			uth->uu_network_marks &= ~unpop;
		}
	}

	return __unsafe_forge_single(net_thread_marks_t, (base + unpop));
}

__private_extern__ void
net_thread_marks_pop(net_thread_marks_t popx)
{
	static const char *__unsafe_indexable const base = (const void*__single)&net_thread_marks_base;
	const ptrdiff_t pop = (const char *)popx - (const char *)base;
	if (pop != 0) {
		static const ptrdiff_t ones = (ptrdiff_t)(u_int32_t)~0U;
		struct uthread *uth = current_uthread();

		VERIFY((pop & ones) == pop);
		VERIFY((ptrdiff_t)(uth->uu_network_marks & pop) == pop);
		uth->uu_network_marks &= ~pop;
	}
}

__private_extern__ void
net_thread_unmarks_pop(net_thread_marks_t unpopx)
{
	static const char *__unsafe_indexable const base = (const void*__single)&net_thread_marks_base;
	ptrdiff_t unpop = (const char *)unpopx - (const char *)base;

	if (unpop != 0) {
		static const ptrdiff_t ones = (ptrdiff_t)(u_int32_t)~0U;
		struct uthread *uth = current_uthread();

		VERIFY((unpop & ones) == unpop);
		VERIFY((ptrdiff_t)(uth->uu_network_marks & unpop) == 0);
		uth->uu_network_marks |= (u_int32_t)unpop;
	}
}


__private_extern__ u_int32_t
net_thread_is_marked(u_int32_t check)
{
	if (check != 0) {
		struct uthread *uth = current_uthread();
		return uth->uu_network_marks & check;
	} else {
		return 0;
	}
}

__private_extern__ u_int32_t
net_thread_is_unmarked(u_int32_t check)
{
	if (check != 0) {
		struct uthread *uth = current_uthread();
		return ~uth->uu_network_marks & check;
	} else {
		return 0;
	}
}
