/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 7, 2025.
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
#if HAVE_GNU_REGEX
#define __USE_GNU 1
#include <regex.h>
#define PATTERN_TYPE          struct re_pattern_buffer *
#define SET_NULL_PATTERN(name)   name = NULL
#endif

#if HAVE_POSIX_REGCOMP
#include <regex.h>
#ifdef REG_EXTENDED
extern int less_is_more;
#define REGCOMP_FLAG    (less_is_more ? 0 : REG_EXTENDED)
#else
#define REGCOMP_FLAG    0
#endif
#define PATTERN_TYPE          regex_t *
#define SET_NULL_PATTERN(name)   name = NULL
#endif

#if HAVE_PCRE
#include <pcre.h>
#define PATTERN_TYPE          pcre *
#define SET_NULL_PATTERN(name)   name = NULL
#endif

#if HAVE_PCRE2
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#define PATTERN_TYPE          pcre2_code *
#define SET_NULL_PATTERN(name)   name = NULL
#endif

#if HAVE_RE_COMP
char *re_comp LESSPARAMS ((char*));
int re_exec LESSPARAMS ((char*));
#define PATTERN_TYPE          int
#define SET_NULL_PATTERN(name)   name = 0
#endif

#if HAVE_REGCMP
char *regcmp LESSPARAMS ((char*));
char *regex LESSPARAMS ((char**, char*));
extern char *__loc1;
#define PATTERN_TYPE          char **
#define SET_NULL_PATTERN(name)   name = NULL
#endif

#if HAVE_V8_REGCOMP
#include "regexp.h"
extern int reg_show_error;
#define PATTERN_TYPE          struct regexp *
#define SET_NULL_PATTERN(name)   name = NULL
#endif

#if NO_REGEX
#define PATTERN_TYPE          void *
#define SET_NULL_PATTERN(name)   
#endif
