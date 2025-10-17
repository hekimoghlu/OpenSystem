/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 11, 2023.
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
#define _GLOB_H_

#include <sys/cdefs.h>
#include <sys/types.h>

struct dirent;
struct stat;

typedef struct {
  size_t gl_pathc;	/* Count of total paths so far. */
  size_t gl_matchc;	/* Count of paths matching pattern. */
  size_t gl_offs;		/* Reserved at beginning of gl_pathv. */
  int gl_flags;		/* Copy of flags parameter to glob. */

  /** List of paths matching pattern. */
  char* _Nullable * _Nullable gl_pathv;

  /** Copy of `__error_callback` parameter to glob. */
  int (* _Nullable gl_errfunc)(const char* _Nonnull __failure_path, int __failure_errno);

  /** Called instead of closedir() when GLOB_ALTDIRFUNC flag is specified. */
  void (* _Nullable gl_closedir)(void* _Nonnull);
  /** Called instead of readdir() when GLOB_ALTDIRFUNC flag is specified. */
  struct dirent* _Nullable (* _Nonnull gl_readdir)(void* _Nonnull);
  /** Called instead of opendir() when GLOB_ALTDIRFUNC flag is specified. */
  void* _Nullable (* _Nonnull gl_opendir)(const char* _Nonnull);
  /** Called instead of lstat() when GLOB_ALTDIRFUNC flag is specified. */
  int (* _Nullable gl_lstat)(const char* _Nonnull, struct stat* _Nonnull);
  /** Called instead of stat() when GLOB_ALTDIRFUNC flag is specified. */
  int (* _Nullable gl_stat)(const char* _Nonnull, struct stat* _Nonnull);
} glob_t;

/* Believed to have been introduced in 1003.2-1992 */
#define GLOB_APPEND	0x0001	/* Append to output from previous call. */
#define GLOB_DOOFFS	0x0002	/* Prepend `gl_offs` null pointers (leaving space for exec, say). */
#define GLOB_ERR	0x0004	/* Return on error. */
#define GLOB_MARK	0x0008	/* Append "/" to the names of returned directories. */
#define GLOB_NOCHECK	0x0010	/* Return pattern itself if nothing matches. */
#define GLOB_NOSORT	0x0020	/* Don't sort. */
#define GLOB_NOESCAPE	0x2000	/* Disable backslash escaping. */

/* Error values returned by glob(3) */
#define GLOB_NOSPACE	(-1)	/* Malloc call failed. */
#define GLOB_ABORTED	(-2)	/* Unignored error. */
#define GLOB_NOMATCH	(-3)	/* No match and GLOB_NOCHECK was not set. */

#if __USE_BSD
#define GLOB_ALTDIRFUNC	0x0040	/* Use alternately specified directory funcs. */
#define GLOB_BRACE	0x0080	/* Expand braces like csh. */
#define GLOB_MAGCHAR	0x0100	/* Set in `gl_flags` if the pattern had globbing characters. */
#define GLOB_NOMAGIC	0x0200	/* GLOB_NOCHECK without magic chars (csh). */
#define GLOB_QUOTE	0x0400	/* Quote special chars with \. */
#define GLOB_TILDE	0x0800	/* Expand tilde names from the passwd file. */
#define GLOB_LIMIT	0x1000	/* limit number of returned paths */
#endif

__BEGIN_DECLS


#if __BIONIC_AVAILABILITY_GUARD(28)
int glob(const char* _Nonnull __pattern, int __flags, int (* _Nullable __error_callback)(const char* _Nonnull __failure_path, int __failure_errno), glob_t* _Nonnull __result_ptr) __INTRODUCED_IN(28);
void globfree(glob_t* _Nonnull __result_ptr) __INTRODUCED_IN(28);
#endif /* __BIONIC_AVAILABILITY_GUARD(28) */


__END_DECLS

#endif
