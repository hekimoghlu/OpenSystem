/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 6, 2023.
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
#pragma prototyped
/*
 * ast POSIX wait/exit support
 */

#ifndef _WAIT_H
#define _WAIT_H

#include <ast.h>
#include <ast_wait.h>

#if _sys_wait
#if defined(__STDPP__directive) && defined(__STDPP__hide)
__STDPP__directive pragma pp:hide wait waitpid
#else
#define wait		______wait
#define waitpid		______waitpid
#endif
#include <sys/wait.h>
#if defined(__STDPP__directive) && defined(__STDPP__hide)
__STDPP__directive pragma pp:nohide wait waitpid
#else
#undef	wait
#undef	waitpid
#endif
#endif

#ifndef WNOHANG
#define WNOHANG		1
#endif

#ifndef WUNTRACED
#define WUNTRACED	2
#endif

#if !_ok_wif
#undef	WIFEXITED
#undef	WEXITSTATUS
#undef	WIFSIGNALED
#undef	WTERMSIG
#undef	WIFSTOPPED
#undef	WSTOPSIG
#undef	WTERMCORE
#endif

#ifndef WIFEXITED
#define WIFEXITED(x)	(!((x)&((1<<(EXIT_BITS-1))-1)))
#endif

#ifndef WEXITSTATUS
#define WEXITSTATUS(x)	(((x)>>EXIT_BITS)&((1<<EXIT_BITS)-1))
#endif

#ifndef WIFSIGNALED
#define WIFSIGNALED(x)	(((x)&((1<<(EXIT_BITS-1))-1))!=0)
#endif

#ifndef WTERMSIG
#define WTERMSIG(x)	((x)&((1<<(EXIT_BITS-1))-1))
#endif

#ifndef WIFSTOPPED
#define WIFSTOPPED(x)	(((x)&((1<<EXIT_BITS)-1))==((1<<(EXIT_BITS-1))-1))
#endif

#ifndef WSTOPSIG
#define WSTOPSIG(x)	WEXITSTATUS(x)
#endif

#ifndef WTERMCORE
#define WTERMCORE(x)	((x)&(1<<(EXIT_BITS-1)))
#endif

extern pid_t		wait(int*);
extern pid_t		waitpid(pid_t, int*, int);

#endif
