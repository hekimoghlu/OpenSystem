/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 23, 2023.
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
#ifndef _NEXT_POSIX_H
#define _NEXT_POSIX_H

#ifdef HAVE_NEXT
#include <sys/dir.h>

/* NGROUPS_MAX is behind -lposix.  Use the BSD version which is NGROUPS */
#undef NGROUPS_MAX
#define NGROUPS_MAX NGROUPS

/* NeXT's readdir() is BSD (struct direct) not POSIX (struct dirent) */
#define dirent direct

/* Swap out NeXT's BSD wait() for a more POSIX compliant one */
pid_t posix_wait(int *);
#define wait(a) posix_wait(a)

/* #ifdef wrapped functions that need defining for clean compiling */
pid_t getppid(void);
void vhangup(void);
int innetgr(const char *, const char *, const char *, const char *);

/* TERMCAP */
int tcgetattr(int, struct termios *);
int tcsetattr(int, int, const struct termios *);
int tcsetpgrp(int, pid_t);
speed_t cfgetospeed(const struct termios *);
speed_t cfgetispeed(const struct termios *);
int cfsetospeed(struct termios *, int);
int cfsetispeed(struct termios *, int);
#endif /* HAVE_NEXT */
#endif /* _NEXT_POSIX_H */
