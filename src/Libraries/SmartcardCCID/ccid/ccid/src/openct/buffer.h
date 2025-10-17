/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 15, 2025.
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
#ifndef OPENCT_BUFFER_H
#define OPENCT_BUFFER_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

typedef struct ct_buf {
	unsigned char *		base;
	unsigned int		head, tail, size;
	unsigned int		overrun;
} ct_buf_t;

extern void		ct_buf_init(ct_buf_t *, void *, size_t);
extern void		ct_buf_set(ct_buf_t *, void *, size_t);
extern int		ct_buf_get(ct_buf_t *, void *, size_t);
extern int		ct_buf_put(ct_buf_t *, const void *, size_t);
extern int		ct_buf_putc(ct_buf_t *, int);
extern unsigned int	ct_buf_avail(ct_buf_t *);
extern void *		ct_buf_head(ct_buf_t *);

#ifdef __cplusplus
}
#endif

#endif /* OPENCT_BUFFER_H */
