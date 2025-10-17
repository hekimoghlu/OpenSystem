/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 2, 2025.
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

#include "libc_private.h"

extern pid_t __fork(void);

static void (*_libSystem_atfork_prepare)(void) = 0;
static void (*_libSystem_atfork_parent)(void) = 0;
static void (*_libSystem_atfork_child)(void) = 0;

__private_extern__
void _libc_fork_init(const struct _libc_functions *funcs)
{
	_libSystem_atfork_prepare = funcs->atfork_prepare;
	_libSystem_atfork_parent = funcs->atfork_parent;
	_libSystem_atfork_child = funcs->atfork_child;
}

/*
 * fork stub
 */
pid_t
fork(void)
{
	int ret;
	
	_libSystem_atfork_prepare();
	// Reader beware: this __fork() call is yet another wrapper around the actual syscall
	// and lives inside libsyscall. The fork syscall needs some cuddling by asm before it's
	// allowed to see the big wide C world.
	ret = __fork();
	if (-1 == ret)
	{
		// __fork already set errno for us
		_libSystem_atfork_parent();
		return ret;
	}
	
	if (0 == ret)
	{
		// We're the child in this part.
		_libSystem_atfork_child();
		return 0;
	}
	
	_libSystem_atfork_parent();
	return ret;
}

