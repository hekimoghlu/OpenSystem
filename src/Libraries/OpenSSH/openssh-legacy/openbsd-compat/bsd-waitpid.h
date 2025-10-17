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
#ifndef _BSD_WAITPID_H
#define _BSD_WAITPID_H

#ifndef HAVE_WAITPID
/* Clean out any potental issues */
#undef WIFEXITED
#undef WIFSTOPPED
#undef WIFSIGNALED

/* Define required functions to mimic a POSIX look and feel */
#define _W_INT(w)	(*(int*)&(w))	/* convert union wait to int */
#define WIFEXITED(w)	(!((_W_INT(w)) & 0377))
#define WIFSTOPPED(w)	((_W_INT(w)) & 0100)
#define WIFSIGNALED(w)	(!WIFEXITED(w) && !WIFSTOPPED(w))
#define WEXITSTATUS(w)	(int)(WIFEXITED(w) ? ((_W_INT(w) >> 8) & 0377) : -1)
#define WTERMSIG(w)	(int)(WIFSIGNALED(w) ? (_W_INT(w) & 0177) : -1)
#define WCOREFLAG	0x80
#define WCOREDUMP(w) 	((_W_INT(w)) & WCOREFLAG)

/* Prototype */
pid_t waitpid(int, int *, int);

#endif /* !HAVE_WAITPID */
#endif /* _BSD_WAITPID_H */
