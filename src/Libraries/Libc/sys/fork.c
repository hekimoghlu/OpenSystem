/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 19, 2022.
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
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <TargetConditionals.h>
#include <stdbool.h>

#include "_libc_init.h" // for libc_atfork_helper

extern pid_t __fork(void);
extern pid_t __vfork(void);

static void (*_libSystem_atfork_prepare)(void) = 0;
static void (*_libSystem_atfork_parent)(void) = 0;
static void (*_libSystem_atfork_child)(void) = 0;
static void (*_libSystem_atfork_prepare_v2)(unsigned int flags, ...) = 0;
static void (*_libSystem_atfork_parent_v2)(unsigned int flags, ...) = 0;
static void (*_libSystem_atfork_child_v2)(unsigned int flags, ...) = 0;

__private_extern__
void _libc_fork_init(const struct _libc_functions *funcs)
{
	if (funcs->version >= 2) {
		_libSystem_atfork_prepare_v2 = funcs->atfork_prepare_v2;
		_libSystem_atfork_parent_v2 = funcs->atfork_parent_v2;
		_libSystem_atfork_child_v2 = funcs->atfork_child_v2;
	} else {
		_libSystem_atfork_prepare = funcs->atfork_prepare;
		_libSystem_atfork_parent = funcs->atfork_parent;
		_libSystem_atfork_child = funcs->atfork_child;
	}
}

static inline __attribute__((always_inline))
pid_t
_do_fork(bool libsystem_atfork_handlers_only)
{
	int ret;

	int flags = libsystem_atfork_handlers_only ? LIBSYSTEM_ATFORK_HANDLERS_ONLY_FLAG : 0;

	if (_libSystem_atfork_prepare_v2) {
		_libSystem_atfork_prepare_v2(flags);
	} else {
		_libSystem_atfork_prepare();
	}
	// Reader beware: this __fork() call is yet another wrapper around the actual syscall
	// and lives inside libsyscall. The fork syscall needs some cuddling by asm before it's
	// allowed to see the big wide C world.
	ret = __fork();
	if (-1 == ret)
	{
		// __fork already set errno for us
		if (_libSystem_atfork_parent_v2) {
			_libSystem_atfork_parent_v2(flags);
		} else {
			_libSystem_atfork_parent();
		}
		return ret;
	}

	if (0 == ret)
	{
		// We're the child in this part.
		if (_libSystem_atfork_child_v2) {
			_libSystem_atfork_child_v2(flags);
		} else {
			_libSystem_atfork_child();
		}
		return 0;
	}

	if (_libSystem_atfork_parent_v2) {
		_libSystem_atfork_parent_v2(flags);
	} else {
		_libSystem_atfork_parent();
	}
	return ret;
}

pid_t
fork(void)
{
	return _do_fork(false);
}

pid_t
vfork(void)
{
	// vfork() is now just fork().
	// Skip the API pthread_atfork handlers, but do call our own
	// Libsystem_atfork handlers. People are abusing vfork in ways where
	// it matters, e.g. tcsh does all kinds of stuff after the vfork. Sigh.
	return _do_fork(true);
}

