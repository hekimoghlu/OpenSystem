/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 15, 2022.
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
#ifndef _SYS_SBUF_H_
#define _SYS_SBUF_H_

#include <sys/_types.h>
#include <stdarg.h>
#include <stddef.h>

/*
 * Structure definition
 */
struct sbuf {
	char            *s_buf;         /* storage buffer */
	void            *s_unused;      /* binary compatibility. */
	int              s_size;        /* size of storage buffer */
	int              s_len;         /* current length of string */
#define SBUF_FIXEDLEN   0x00000000      /* fixed length buffer (default) */
#define SBUF_AUTOEXTEND 0x00000001      /* automatically extend buffer */
#define SBUF_USRFLAGMSK 0x0000ffff      /* mask of flags the user may specify */
#define SBUF_DYNAMIC    0x00010000      /* s_buf must be freed */
#define SBUF_FINISHED   0x00020000      /* set by sbuf_finish() */
#define SBUF_OVERFLOWED 0x00040000      /* sbuf overflowed */
#define SBUF_DYNSTRUCT  0x00080000      /* sbuf must be freed */
	int              s_flags;       /* flags */
};

__BEGIN_DECLS
#ifdef KERNEL_PRIVATE
struct sbuf     *sbuf_new(struct sbuf *, char *, int, int);
void             sbuf_clear(struct sbuf *);
int              sbuf_setpos(struct sbuf *, int);
int              sbuf_bcat(struct sbuf *, const void *, size_t);
int              sbuf_bcpy(struct sbuf *, const void *, size_t);
int              sbuf_cat(struct sbuf *, const char *);
int              sbuf_cpy(struct sbuf *, const char *);
int              sbuf_printf(struct sbuf *, const char *, ...) __printflike(2, 3);
int              sbuf_vprintf(struct sbuf *, const char *, va_list) __printflike(2, 0);
int              sbuf_putc(struct sbuf *, int);
int              sbuf_trim(struct sbuf *);
int              sbuf_overflowed(struct sbuf *);
void             sbuf_finish(struct sbuf *);
char            *sbuf_data(struct sbuf *);
int              sbuf_len(struct sbuf *);
int              sbuf_done(struct sbuf *);
void             sbuf_delete(struct sbuf *);
#endif

__END_DECLS

#endif
