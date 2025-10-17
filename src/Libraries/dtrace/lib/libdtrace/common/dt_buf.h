/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 23, 2025.
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
/*
 * Copyright 2005 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

#ifndef	_DT_BUF_H
#define	_DT_BUF_H

#ifdef	__cplusplus
extern "C" {
#endif

#include <dtrace.h>

typedef struct dt_buf {
	const char *dbu_name;	/* string name for debugging */
	uchar_t *dbu_buf;	/* buffer base address */
	uchar_t *dbu_ptr;	/* current buffer location */
	size_t dbu_len;		/* buffer size in bytes */
	int dbu_err;		/* errno value if error */
	int dbu_resizes;	/* number of resizes */
} dt_buf_t;

extern void dt_buf_create(dtrace_hdl_t *, dt_buf_t *, const char *, size_t);
extern void dt_buf_destroy(dtrace_hdl_t *, dt_buf_t *);
extern void dt_buf_reset(dtrace_hdl_t *, dt_buf_t *);

extern void dt_buf_write(dtrace_hdl_t *, dt_buf_t *,
    const void *, size_t, size_t);

extern void dt_buf_concat(dtrace_hdl_t *, dt_buf_t *,
    const dt_buf_t *, size_t);

extern size_t dt_buf_offset(const dt_buf_t *, size_t);
extern size_t dt_buf_len(const dt_buf_t *);

extern int dt_buf_error(const dt_buf_t *);
extern void *dt_buf_ptr(const dt_buf_t *);

extern void *dt_buf_claim(dtrace_hdl_t *, dt_buf_t *);

#ifdef	__cplusplus
}
#endif

#endif	/* _DT_BUF_H */
