/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 4, 2022.
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
#ifndef _REGEXP
#define _REGEXP 1

#define NSUBEXP  10
typedef struct regexp {
        char *startp[NSUBEXP];
        char *endp[NSUBEXP];
        char regstart;          /* Internal use only. */
        char reganch;           /* Internal use only. */
        char *regmust;          /* Internal use only. */
        int regmlen;            /* Internal use only. */
        char program[1];        /* Unwarranted chumminess with compiler. */
} regexp;

#if defined(__STDC__) || defined(__cplusplus)
#   define _ANSI_ARGS_(x)       x
#else
#   define _ANSI_ARGS_(x)       ()
#endif

extern regexp *regcomp _ANSI_ARGS_((char *exp));
extern int regexec _ANSI_ARGS_((regexp *prog, char *string));
extern int regexec2 _ANSI_ARGS_((regexp *prog, char *string, int notbol));
extern void regsub _ANSI_ARGS_((regexp *prog, char *source, char *dest));
extern void regerror _ANSI_ARGS_((char *msg));

#endif /* REGEXP */
