/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 14, 2025.
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
/* Id: misc.h,v 1.9 2006/04/06 14:00:06 manubsd Exp */

/*
 * Copyright (C) 1995, 1996, 1997, and 1998 WIDE Project.
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the project nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE PROJECT AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE PROJECT OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#ifndef _MISC_H
#define _MISC_H

#define BIT2STR(b) bit2str(b, sizeof(b)<<3)

#ifdef HAVE_FUNC_MACRO
#define LOCATION        debug_location(__FILE__, __LINE__, __func__)
#else
#define LOCATION        debug_location(__FILE__, __LINE__, NULL)
#endif

extern int hexdump (void *, size_t);
extern char *bit2str (int, int);
extern void *get_newbuf (void *, size_t);
extern const char *debug_location (const char *, int, const char *);
extern int getfsize (char *);
struct timeval;
extern double timedelta (struct timeval *, struct timeval *);
char *strdup (const char *);
extern char* binsanitize (char*, size_t);

#define RACOON_TAILQ_FOREACH_REVERSE(var, head, headname ,field)	\
  TAILQ_FOREACH_REVERSE(var, head, field, headname)

#define STRDUP_FATAL(x) if (x == NULL) {			\
	plog(ASL_LEVEL_ERR, "strdup failed\n");	\
	exit(1);						\
}

#include "libpfkey.h"

#define remainingsize(string_buffer_sizeof, filled_str) (string_buffer_sizeof - strlen(filled_str) - 1)
#define remainingsize_opt(string_buffer_sizeof, filled_strlen) (string_buffer_sizeof - filled_strlen - 1)
#define remainingsizeof(string_buffer) (sizeof(string_buffer) - strlen(string_buffer) - 1)
#define remainingsizeof_opt(string_buffer, filled_strlen) (sizeof(string_buffer) - filled_strlen - 1)

#endif /* _MISC_H */
