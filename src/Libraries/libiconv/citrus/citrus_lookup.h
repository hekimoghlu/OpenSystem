/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 13, 2024.
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
/* $NetBSD: citrus_lookup.h,v 1.2 2004/07/21 14:16:34 tshiozak Exp $ */

/*-
 * SPDX-License-Identifier: BSD-2-Clause
 *
 * Copyright (c)2003 Citrus Project,
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
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#ifndef _CITRUS_LOOKUP_H_
#define _CITRUS_LOOKUP_H_

#define _CITRUS_LOOKUP_CASE_SENSITIVE	0
#define _CITRUS_LOOKUP_CASE_IGNORE	1

struct _citrus_lookup;
#ifdef __APPLE__
struct _region;
#endif
__BEGIN_DECLS
char		*_citrus_lookup_simple(const char *, const char *, char *,
		    size_t, int);
int		 _citrus_lookup_seq_open(struct _citrus_lookup **,
		    const char *, int);
void		 _citrus_lookup_seq_rewind(struct _citrus_lookup *);
int		 _citrus_lookup_seq_next(struct _citrus_lookup *,
			    struct _region *, struct _region *);
int		 _citrus_lookup_seq_lookup(struct _citrus_lookup *,
		    const char *, struct _region *);
int		 _citrus_lookup_get_number_of_entries(struct _citrus_lookup *);
void		 _citrus_lookup_seq_close(struct _citrus_lookup *);
__END_DECLS

static __inline const char *
_citrus_lookup_alias(const char *path, const char *key, char *buf, size_t n,
    int ignore_case)
{
	const char *ret;

	ret = _citrus_lookup_simple(path, key, buf, n, ignore_case);
	if (ret == NULL)
		ret = key;

	return (ret);
}

#endif
