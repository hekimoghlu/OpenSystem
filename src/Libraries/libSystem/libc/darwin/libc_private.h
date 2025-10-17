/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 7, 2023.
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
#ifndef __LIBC_PRIVATE_H__
#define __LIBC_PRIVATE_H__

#include <sys/cdefs.h>
#include <Availability.h>
#include <stddef.h>

__BEGIN_DECLS

struct _libc_functions {
	unsigned long version;
	void (*atfork_prepare)(void); // version 1
	void (*atfork_parent)(void); // version 1
	void (*atfork_child)(void); // version 1
	char *(*dirhelper)(int, char *, size_t); // version 1
};

struct ProgramVars; // forward reference

__deprecated_msg("use _libc_initializer()")
extern void
__libc_init(const struct ProgramVars *vars,
	void (*atfork_prepare)(void),
	void (*atfork_parent)(void),
	void (*atfork_child)(void),
	const char *apple[]);

__OSX_AVAILABLE_STARTING(__MAC_10_10, __IPHONE_8_0)
extern void
_libc_initializer(const struct _libc_functions *funcs,
	const char *envp[],
	const char *apple[],
	const struct ProgramVars *vars);

extern void
_libc_fork_child(void);

extern int
_atexit_receipt(void);

__IOS_AVAILABLE(9.0) __OSX_AVAILABLE(10.11)
extern void
abort_report_np(const char *, ...) __dead2 __cold __printflike(1, 2);

__END_DECLS

#endif // __LIBC_PRIVATE_H__
