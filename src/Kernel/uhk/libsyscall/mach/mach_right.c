/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 14, 2022.
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
#include <mach/mach.h>
#include <mach/mach_traps.h>
#include <mach/mach_port.h>
#include <mach/mach_right.h>
#include <mach/mach_right_private.h>


#pragma mark Utilities
#define _mach_assert(__op, __kr) \
	do { \
	        if (kr != KERN_SUCCESS) { \
	                __builtin_trap(); \
	        } \
	} while (0)

#pragma mark API
mach_right_recv_t
mach_right_recv_construct(mach_right_flags_t flags,
    mach_right_send_t *_Nullable sr, uintptr_t ctx)
{
	kern_return_t kr = KERN_FAILURE;
	mach_port_t p = MACH_PORT_NULL;
	mach_port_options_t opts = {
		.flags = MPO_CONTEXT_AS_GUARD,
		.mpl = {
			.mpl_qlimit = MACH_PORT_QLIMIT_BASIC,
		},
	};

	if (flags & MACH_RIGHT_RECV_FLAG_UNGUARDED) {
		opts.flags &= (~MPO_CONTEXT_AS_GUARD);
	}
	if (flags & MACH_RIGHT_RECV_FLAG_STRICT) {
		opts.flags |= MPO_STRICT;
	}
	if (flags & MACH_RIGHT_RECV_FLAG_IMMOVABLE) {
		opts.flags |= MPO_IMMOVABLE_RECEIVE;
	}
	if (sr) {
		opts.flags |= MPO_INSERT_SEND_RIGHT;
	}

	kr = mach_port_construct(mach_task_self(), &opts, ctx, &p);
	_mach_assert("construct recv right", kr);

	if (sr) {
		sr->mrs_name = p;
	}

	return mach_right_recv(p);
}

void
mach_right_recv_destruct(mach_right_recv_t r, mach_right_send_t *s,
    uintptr_t ctx)
{
	kern_return_t kr = KERN_FAILURE;
	mach_port_delta_t srd = 0;

	if (s) {
		if (r.mrr_name != s->mrs_name) {
			__builtin_trap();
		}

		srd = -1;
	}

	kr = mach_port_destruct(mach_task_self(), r.mrr_name, srd, ctx);
	_mach_assert("destruct recv right", kr);
}

mach_right_send_t
mach_right_send_create(mach_right_recv_t r)
{
	kern_return_t kr = KERN_FAILURE;

	kr = mach_port_insert_right(mach_task_self(), r.mrr_name, r.mrr_name,
	    MACH_MSG_TYPE_MAKE_SEND);
	_mach_assert("create send right", kr);

	return mach_right_send(r.mrr_name);
}

mach_right_send_t
mach_right_send_retain(mach_right_send_t s)
{
	kern_return_t kr = KERN_FAILURE;
	mach_right_send_t rs = MACH_RIGHT_SEND_NULL;

	kr = mach_port_mod_refs(mach_task_self(), s.mrs_name,
	    MACH_PORT_RIGHT_SEND, 1);
	switch (kr) {
	case 0:
		rs = s;
		break;
	case KERN_INVALID_RIGHT:
		rs.mrs_name = MACH_PORT_DEAD;
		break;
	case KERN_INVALID_NAME:
	// mach_port_mod_refs() will return success when given either
	// MACH_PORT_DEAD or MACH_PORT_NULL with send or send-once right
	// operations, so this is always fatal.
	default:
		_mach_assert("retain send right", kr);
	}

	return rs;
}

void
mach_right_send_release(mach_right_send_t s)
{
	kern_return_t kr = KERN_FAILURE;

	kr = mach_port_mod_refs(mach_task_self(), s.mrs_name,
	    MACH_PORT_RIGHT_SEND, -1);
	switch (kr) {
	case 0:
		break;
	case KERN_INVALID_RIGHT:
		kr = mach_port_mod_refs(mach_task_self(), s.mrs_name,
		    MACH_PORT_RIGHT_DEAD_NAME, -1);
		_mach_assert("release dead name", kr);
		break;
	default:
		_mach_assert("release send right", kr);
	}
}

mach_right_send_once_t
mach_right_send_once_create(mach_right_recv_t r)
{
	mach_msg_type_name_t right = 0;
	mach_port_t so = MACH_PORT_NULL;
	kern_return_t kr = mach_port_extract_right(mach_task_self(), r.mrr_name,
	    MACH_MSG_TYPE_MAKE_SEND_ONCE, &so, &right);
	_mach_assert("create send-once right", kr);

	return mach_right_send_once(so);
}

void
mach_right_send_once_consume(mach_right_send_once_t so)
{
	kern_return_t kr = KERN_FAILURE;

	kr = mach_port_mod_refs(mach_task_self(), so.mrso_name,
	    MACH_PORT_RIGHT_SEND_ONCE, -1);
	switch (kr) {
	case 0:
		break;
	case KERN_INVALID_RIGHT:
		kr = mach_port_mod_refs(mach_task_self(), so.mrso_name,
		    MACH_PORT_RIGHT_DEAD_NAME, -1);
		_mach_assert("release dead name", kr);
		break;
	default:
		_mach_assert("consume send-once right", kr);
	}
}
