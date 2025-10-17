/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 20, 2022.
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
 * _libc_initializer() is called from libSystem_initializer()
 */

#include <crt_externs.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <machine/cpu_capabilities.h>
#include <TargetConditionals.h>
#include <_simple.h>

#include "_libc_init.h"

extern void _program_vars_init(const struct ProgramVars *vars);
extern void _libc_fork_init(const struct _libc_functions *funcs);
extern void _arc4_init(void);
extern void __atexit_init(void);
extern void __confstr_init(const struct _libc_functions *funcs);
extern void _init_clock_port(void);
extern void __chk_init(void);
extern void __xlocale_init(void);
extern void __guard_setup(const char *apple[]);
extern void _subsystem_init(const char *apple[]);
extern void __stdio_init(void);

void
_libc_initializer(const struct _libc_functions *funcs,
	const char *envp[],
	const char *apple[],
	const struct ProgramVars *vars)
{
	_program_vars_init(vars);
	_libc_fork_init(funcs);
	__confstr_init(funcs);
	__atexit_init();
	_init_clock_port();
	__chk_init();
	__xlocale_init();
	__guard_setup(apple);
	_subsystem_init(apple);
	__stdio_init();
}


void
__libc_init(const struct ProgramVars *vars,
	void (*atfork_prepare)(void),
	void (*atfork_parent)(void),
	void (*atfork_child)(void),
	const char *apple[])
{
	const struct _libc_functions funcs = {
		.version = 1,
		.atfork_prepare = atfork_prepare,
		.atfork_parent = atfork_parent,
		.atfork_child = atfork_child,
		.dirhelper = NULL,
	};
	
	return _libc_initializer(&funcs, NULL, apple, vars);
}
