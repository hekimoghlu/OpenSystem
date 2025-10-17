/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 12, 2022.
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
#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#ifndef HAVE_STDLIB_H
char *malloc _((size_t));
char *realloc _((void *, size_t));
char *calloc _((size_t, size_t));
#endif

#if !(defined(USES_TERMCAP_H) || defined(USES_TERM_H))
/*
 * These prototypes are only used where we don't have the
 * headers.  In some cases they need tweaking.
 * TBD: we'd much prefer to get hold of the header where
 * these are defined.
 */
#ifdef _AIX
#define TC_CONST const
#else
#define TC_CONST
#endif
extern int tgetent _((char *bp, TC_CONST char *name));
extern int tgetnum _((char *id));
extern int tgetflag _((char *id));
extern char *tgetstr _((char *id, char **area));
extern int tputs _((TC_CONST char *cp, int affcnt, int (*outc) (int)));
#undef TC_CONST
#endif

/*
 * Some systems that do have termcap headers nonetheless don't
 * declare tgoto, so we detect if that is missing separately.
 */
#ifdef TGOTO_PROTO_MISSING
char *tgoto(const char *cap, int col, int row);
#endif

/* MISSING PROTOTYPES FOR VARIOUS OPERATING SYSTEMS */

#if defined(__hpux) && defined(_HPUX_SOURCE) && !defined(_XPG4_EXTENDED)
# define SELECT_ARG_2_T int *
#else
# define SELECT_ARG_2_T fd_set *
#endif

#ifdef __osf__
char *mktemp _((char *));
#endif

#if defined(__osf__) && defined(__alpha) && defined(__GNUC__)
/* Digital cc does not need these prototypes, gcc does need them */
# ifndef HAVE_IOCTL_PROTO
int ioctl _((int d, unsigned long request, void *argp));
# endif
# ifndef HAVE_MKNOD_PROTO
int mknod _((const char *pathname, int mode, dev_t device));
# endif
int nice _((int increment));
int select _((int nfds, fd_set * readfds, fd_set * writefds, fd_set * exceptfds, struct timeval *timeout));
#endif

#if defined(DGUX) && defined(__STDC__)
/* Just plain missing. */
extern int getrlimit _((int resource, struct rlimit *rlp));
extern int setrlimit _((int resource, const struct rlimit *rlp));
extern int getrusage _((int who, struct rusage *rusage));
extern int gettimeofday _((struct timeval *tv, struct timezone *tz));
extern int wait3 _((union wait *wait_status, int options, struct rusage *rusage));
extern int getdomainname _((char *name, int maxlength));
extern int select _((int nfds, fd_set * readfds, fd_set * writefds, fd_set * exceptfds, struct timeval *timeout));
#endif /* DGUX and __STDC__ */

#ifdef __NeXT__
extern pid_t getppid(void);
#endif

#if defined(__sun__) && !defined(__SVR4)  /* SunOS */
extern char *strerror _((int errnum));
#endif

/**************************************************/
/*** prototypes for functions built in compat.c ***/
#ifndef HAVE_STRSTR
extern char *strstr _((const char *s, const char *t));
#endif

#ifndef HAVE_GETHOSTNAME
extern int gethostname _((char *name, size_t namelen));
#endif

#ifndef HAVE_GETTIMEOFDAY
extern int gettimeofday _((struct timeval *tv, struct timezone *tz));
#endif

#ifndef HAVE_DIFFTIME
extern double difftime _((time_t t2, time_t t1));
#endif

#ifndef HAVE_STRERROR
extern char *strerror _((int errnum));
#endif

/*** end of prototypes for functions in compat.c ***/
/***************************************************/

#ifndef HAVE_MEMMOVE
extern void bcopy _((const void *, void *, size_t));
#endif

#if defined(__APPLE__) && TARGET_OS_OSX
const char *check_managed_config(const char *managed_config, const char *default_config);
#endif // __APPLE__
