/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 4, 2023.
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
#ifndef _GLOB_H_
#define	_GLOB_H_

#include <_types.h>
#include <sys/cdefs.h>
#include <_bounds.h>
#include <Availability.h>
#include <sys/_types/_size_t.h>

_LIBC_SINGLE_BY_DEFAULT()

#if !defined(_POSIX_C_SOURCE) || defined(_DARWIN_C_SOURCE)
struct dirent;
struct stat;
#endif /* (!_POSIX_C_SOURCE || _DARWIN_C_SOURCE) */
typedef struct {
	size_t gl_pathc;	/* Count of total paths so far. */
	int gl_matchc;		/* Count of paths matching pattern. */
	size_t gl_offs;		/* Reserved at beginning of gl_pathv. */
	int gl_flags;		/* Copy of flags parameter to glob. */
	char *_LIBC_CSTR *_LIBC_COUNT(gl_matchc)	gl_pathv; /* List of paths matching pattern. */
				/* Copy of errfunc parameter to glob. */
#ifdef __BLOCKS__
	union {
#endif /* __BLOCKS__ */
		int (*gl_errfunc)(const char *, int);
#ifdef __BLOCKS__
		int (^gl_errblk)(const char *, int);
	};
#endif /* __BLOCKS__ */

	/*
	 * Alternate filesystem access methods for glob; replacement
	 * versions of closedir(3), readdir(3), opendir(3), stat(2)
	 * and lstat(2).
	 */
	void (*gl_closedir)(void *);
#if !defined(_POSIX_C_SOURCE) || defined(_DARWIN_C_SOURCE)
	struct dirent *(*gl_readdir)(void *);
#else /* (_POSIX_C_SOURCE && !_DARWIN_C_SOURCE) */
	void *(*gl_readdir)(void *);
#endif /* (!_POSIX_C_SOURCE || _DARWIN_C_SOURCE) */
	void *(*gl_opendir)(const char *);
#if !defined(_POSIX_C_SOURCE) || defined(_DARWIN_C_SOURCE)
	int (*gl_lstat)(const char *, struct stat *);
	int (*gl_stat)(const char *, struct stat *);
#else /* (_POSIX_C_SOURCE && !_DARWIN_C_SOURCE) */
	int (*gl_lstat)(const char *, void *);
	int (*gl_stat)(const char *, void *);
#endif /* (!_POSIX_C_SOURCE || _DARWIN_C_SOURCE) */
} glob_t;

/* Believed to have been introduced in 1003.2-1992 */
#define	GLOB_APPEND	0x0001	/* Append to output from previous call. */
#define	GLOB_DOOFFS	0x0002	/* Use gl_offs. */
#define	GLOB_ERR	0x0004	/* Return on error. */
#define	GLOB_MARK	0x0008	/* Append / to matching directories. */
#define	GLOB_NOCHECK	0x0010	/* Return pattern itself if nothing matches. */
#define	GLOB_NOSORT	0x0020	/* Don't sort. */
#define	GLOB_NOESCAPE	0x2000	/* Disable backslash escaping. */

/* Error values returned by glob(3) */
#define	GLOB_NOSPACE	(-1)	/* Malloc call failed. */
#define	GLOB_ABORTED	(-2)	/* Unignored error. */
#define	GLOB_NOMATCH	(-3)	/* No match and GLOB_NOCHECK was not set. */
#define	GLOB_NOSYS	(-4)	/* Obsolete: source comptability only. */

#define	GLOB_ALTDIRFUNC	0x0040	/* Use alternately specified directory funcs. */
#define	GLOB_BRACE	0x0080	/* Expand braces ala csh. */
#define	GLOB_MAGCHAR	0x0100	/* Pattern had globbing characters. */
#define	GLOB_NOMAGIC	0x0200	/* GLOB_NOCHECK without magic chars (csh). */
#define	GLOB_QUOTE	0x0400	/* Quote special chars with \. */
#define	GLOB_TILDE	0x0800	/* Expand tilde names from the passwd file. */
#define	GLOB_LIMIT	0x1000	/* limit number of returned paths */
#ifdef __BLOCKS__
#define	_GLOB_ERR_BLOCK	0x80000000 /* (internal) error callback is a block */
#endif /* __BLOCKS__ */

/* source compatibility, these are the old names */
#define GLOB_MAXPATH	GLOB_LIMIT
#define	GLOB_ABEND	GLOB_ABORTED

__BEGIN_DECLS
//Begin-Libc
#ifndef LIBC_ALIAS_GLOB
//End-Libc
int	glob(const char * __restrict, int, int (*)(const char *, int), 
	     glob_t * __restrict) __DARWIN_INODE64(glob);
//Begin-Libc
#else /* LIBC_ALIAS_GLOB */
int	glob(const char * __restrict, int, int (*)(const char *, int), 
	     glob_t * __restrict) LIBC_INODE64(glob);
#endif /* !LIBC_ALIAS_GLOB */
//End-Libc
#ifdef __BLOCKS__
#if __has_attribute(noescape)
#define __glob_noescape __attribute__((__noescape__))
#else
#define __glob_noescape
#endif
//Begin-Libc
#ifndef LIBC_ALIAS_GLOB_B
//End-Libc
int	glob_b(const char * __restrict, int, int (^)(const char *, int) __glob_noescape,
	     glob_t * __restrict) __DARWIN_INODE64(glob_b) __OSX_AVAILABLE_STARTING(__MAC_10_6, __IPHONE_3_2);
//Begin-Libc
#else /* LIBC_ALIAS_GLOB_B */
int	glob_b(const char * __restrict, int, int (^)(const char *, int) __glob_noescape,
	     glob_t * __restrict) LIBC_INODE64(glob_b);
#endif /* !LIBC_ALIAS_GLOB_B */
//End-Libc
#endif /* __BLOCKS__ */
void	globfree(glob_t *);
__END_DECLS

#endif /* !_GLOB_H_ */
