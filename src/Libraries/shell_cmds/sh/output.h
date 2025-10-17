/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 4, 2023.
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
#ifndef OUTPUT_INCL

#include <stdarg.h>
#include <stddef.h>

struct output {
	char *nextc;
	char *bufend;
	char *buf;
	int bufsize;
	short fd;
	short flags;
};

extern struct output output; /* to fd 1 */
extern struct output errout; /* to fd 2 */
extern struct output memout;
extern struct output *out1; /* &memout if backquote, otherwise &output */
extern struct output *out2; /* &memout if backquote with 2>&1, otherwise
			       &errout */

void outcslow(int, struct output *);
void out1str(const char *);
void out1qstr(const char *);
void out2str(const char *);
void out2qstr(const char *);
void outstr(const char *, struct output *);
void outqstr(const char *, struct output *);
void outbin(const void *, size_t, struct output *);
void emptyoutbuf(struct output *);
void flushall(void);
void flushout(struct output *);
void freestdout(void);
int outiserror(struct output *);
void outclearerror(struct output *);
void outfmt(struct output *, const char *, ...) __printflike(2, 3);
void out1fmt(const char *, ...) __printflike(1, 2);
void out2fmt_flush(const char *, ...) __printflike(1, 2);
void fmtstr(char *, int, const char *, ...) __printflike(3, 4);
void doformat(struct output *, const char *, va_list) __printflike(2, 0);
int xwrite(int, const char *, int);

#define outc(c, file)	((file)->nextc == (file)->bufend ? (emptyoutbuf(file), *(file)->nextc++ = (c)) : (*(file)->nextc++ = (c)))
#define out1c(c)	outc(c, out1);
#define out2c(c)	outcslow(c, out2);

#define OUTPUT_INCL
#endif
