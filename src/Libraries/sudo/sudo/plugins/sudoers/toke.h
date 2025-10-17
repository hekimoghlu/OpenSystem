/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 22, 2022.
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
#ifndef SUDOERS_TOKE_H
#define SUDOERS_TOKE_H

struct sudolinebuf {
    char *buf;			/* line buffer */
    size_t size;		/* size of buffer */
    size_t len;			/* used length */
    size_t off;			/* consumed length */
    size_t toke_start;		/* starting column of current token */
    size_t toke_end;		/* ending column of current token */
};
extern const char *sudoers_errstr;
extern struct sudolinebuf sudolinebuf;

bool append(const char *, size_t);
bool fill_args(const char *, size_t, int);
bool fill_cmnd(const char *, size_t);
bool fill(const char *, size_t);
bool ipv6_valid(const char *s);
int sudoers_trace_print(const char *);
void sudoerserrorf(const char *, ...) sudo_printf0like(1, 2);
void sudoerserror(const char *);
bool push_include(const char *, bool);

#ifndef FLEX_SCANNER
extern int (*trace_print)(const char *msg);
#endif

#define LEXTRACE(msg)   do {						\
    if (trace_print != NULL)						\
	(*trace_print)(msg);						\
} while (0);

#endif /* SUDOERS_TOKE_H */
