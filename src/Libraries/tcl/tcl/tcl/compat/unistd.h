/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 23, 2023.
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
#ifndef _UNISTD
#define _UNISTD

#include <sys/types.h>
#ifndef _TCL
#   include "tcl.h"
#endif

#ifndef NULL
#define NULL    0
#endif

/* 
 * Strict POSIX stuff goes here.  Extensions go down below, in the 
 * ifndef _POSIX_SOURCE section.
 */

extern void _exit _ANSI_ARGS_((int status));
extern int access _ANSI_ARGS_((CONST char *path, int mode));
extern int chdir _ANSI_ARGS_((CONST char *path));
extern int chown _ANSI_ARGS_((CONST char *path, uid_t owner, gid_t group));
extern int close _ANSI_ARGS_((int fd));
extern int dup _ANSI_ARGS_((int oldfd));
extern int dup2 _ANSI_ARGS_((int oldfd, int newfd));
extern int execl _ANSI_ARGS_((CONST char *path, ...));
extern int execle _ANSI_ARGS_((CONST char *path, ...));
extern int execlp _ANSI_ARGS_((CONST char *file, ...));
extern int execv _ANSI_ARGS_((CONST char *path, char **argv));
extern int execve _ANSI_ARGS_((CONST char *path, char **argv, char **envp));
extern int execvp _ANSI_ARGS_((CONST char *file, char **argv));
extern pid_t fork _ANSI_ARGS_((void));
extern char *getcwd _ANSI_ARGS_((char *buf, size_t size));
extern gid_t getegid _ANSI_ARGS_((void));
extern uid_t geteuid _ANSI_ARGS_((void));
extern gid_t getgid _ANSI_ARGS_((void));
extern int getgroups _ANSI_ARGS_((int bufSize, int *buffer));
extern pid_t getpid _ANSI_ARGS_((void));
extern uid_t getuid _ANSI_ARGS_((void));
extern int isatty _ANSI_ARGS_((int fd));
extern long lseek _ANSI_ARGS_((int fd, long offset, int whence));
extern int pipe _ANSI_ARGS_((int *fildes));
extern int read _ANSI_ARGS_((int fd, char *buf, size_t size));
extern int setgid _ANSI_ARGS_((gid_t group));
extern int setuid _ANSI_ARGS_((uid_t user));
extern unsigned sleep _ANSI_ARGS_ ((unsigned seconds));
extern char *ttyname _ANSI_ARGS_((int fd));
extern int unlink _ANSI_ARGS_((CONST char *path));
extern int write _ANSI_ARGS_((int fd, CONST char *buf, size_t size));

#ifndef	_POSIX_SOURCE
extern char *crypt _ANSI_ARGS_((CONST char *, CONST char *));
extern int fchown _ANSI_ARGS_((int fd, uid_t owner, gid_t group));
extern int flock _ANSI_ARGS_((int fd, int operation));
extern int ftruncate _ANSI_ARGS_((int fd, unsigned long length));
extern int ioctl _ANSI_ARGS_((int fd, int request, ...));
extern int readlink _ANSI_ARGS_((CONST char *path, char *buf, int bufsize));
extern int setegid _ANSI_ARGS_((gid_t group));
extern int seteuid _ANSI_ARGS_((uid_t user));
extern int setreuid _ANSI_ARGS_((int ruid, int euid));
extern int symlink _ANSI_ARGS_((CONST char *, CONST char *));
extern int ttyslot _ANSI_ARGS_((void));
extern int truncate _ANSI_ARGS_((CONST char *path, unsigned long length));
extern int vfork _ANSI_ARGS_((void));
#endif /* _POSIX_SOURCE */

#endif /* _UNISTD */

