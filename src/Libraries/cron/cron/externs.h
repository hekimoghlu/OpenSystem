/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 16, 2021.
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
#if defined(POSIX) || defined(ATT)
# include <stdlib.h>
# include <unistd.h>
# include <string.h>
# include <dirent.h>
# define DIR_T	struct dirent
# define WAIT_T	int
# define WAIT_IS_INT 1
extern char *tzname[2];
# define TZONE(tm) tzname[(tm).tm_isdst]
#endif

#if defined(UNIXPC)
# undef WAIT_T
# undef WAIT_IS_INT
# define WAIT_T	union wait
#endif

#if defined(POSIX)
# define SIG_T	sig_t
# define TIME_T	time_t
# define PID_T pid_t
#endif

#if defined(ATT)
# define SIG_T	void
# define TIME_T	long
# define PID_T int
#endif

#if !defined(POSIX) && !defined(ATT)
/* classic BSD */
extern	time_t		time();
extern	unsigned	sleep();
extern	struct tm	*localtime();
extern	struct passwd	*getpwnam();
extern	int		errno;
extern	void		perror(), exit(), free();
extern	char		*getenv(), *strcpy(), *strchr(), *strtok();
extern	void		*malloc(), *realloc();
# define SIG_T	void
# define TIME_T	long
# define PID_T int
# define WAIT_T	union wait
# define DIR_T	struct direct
# include <sys/dir.h>
# define TZONE(tm) (tm).tm_zone
#endif

/* getopt() isn't part of POSIX.  some systems define it in <stdlib.h> anyway.
 * of those that do, some complain that our definition is different and some
 * do not.  to add to the misery and confusion, some systems define getopt()
 * in ways that we cannot predict or comprehend, yet do not define the adjunct
 * external variables needed for the interface.
 */
#if (!defined(BSD) || (BSD < 198911)) && !defined(ATT) && !defined(UNICOS)
int	getopt __P((int, char * const *, const char *));
#endif

#if (!defined(BSD) || (BSD < 199103))
extern	char *optarg;
extern	int optind, opterr, optopt;
#endif

#if WAIT_IS_INT
# ifndef WEXITSTATUS
#  define WEXITSTATUS(x) (((x) >> 8) & 0xff)
# endif
# ifndef WTERMSIG
#  define WTERMSIG(x)	((x) & 0x7f)
# endif
# ifndef WCOREDUMP
#  define WCOREDUMP(x)	((x) & 0x80)
# endif
#else /*WAIT_IS_INT*/
# ifndef WEXITSTATUS
#  define WEXITSTATUS(x) ((x).w_retcode)
# endif
# ifndef WTERMSIG
#  define WTERMSIG(x)	((x).w_termsig)
# endif
# ifndef WCOREDUMP
#  define WCOREDUMP(x)	((x).w_coredump)
# endif
#endif /*WAIT_IS_INT*/

#ifndef WIFSIGNALED
#define WIFSIGNALED(x)	(WTERMSIG(x) != 0)
#endif
#ifndef WIFEXITED
#define WIFEXITED(x)	(WTERMSIG(x) == 0)
#endif

#ifdef NEED_STRCASECMP
extern	int		strcasecmp __P((char *, char *));
#endif

#ifdef NEED_STRDUP
extern	char		*strdup __P((char *));
#endif

#ifdef NEED_STRERROR
extern	char		*strerror __P((int));
#endif

#ifdef NEED_FLOCK
extern	int		flock __P((int, int));
# define LOCK_SH 1
# define LOCK_EX 2
# define LOCK_NB 4
# define LOCK_UN 8
#endif

#ifdef NEED_SETSID
extern	int		setsid __P((void));
#endif

#ifdef NEED_GETDTABLESIZE
extern	int		getdtablesize __P((void));
#endif

#ifdef NEED_SETENV
extern	int		setenv __P((char *, char *, int));
#endif

#ifdef NEED_VFORK
extern	PID_T		vfork __P((void));
#endif
