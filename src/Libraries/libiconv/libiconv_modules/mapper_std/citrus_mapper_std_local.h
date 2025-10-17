/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 14, 2024.
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
/*	$NetBSD: citrus_mapper_std_local.h,v 1.3 2006/09/09 14:35:17 tnozaki Exp $	*/

/*-
 * SPDX-License-Identifier: BSD-2-Clause
 *
 * Copyright (c)2003, 2006 Citrus Project,
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

#ifndef _CITRUS_MAPPER_STD_LOCAL_H_
#define _CITRUS_MAPPER_STD_LOCAL_H_

#ifdef __APPLE__
#include <stdbool.h>
#endif

typedef uint32_t (*_citrus_mapper_std_getvalfunc_t)(const void *, uint32_t);

struct _citrus_mapper_std_linear_zone {
	_citrus_index_t		begin;
	_citrus_index_t		end;
	_citrus_index_t		width;
};
struct _citrus_mapper_std_rowcol {
	struct _citrus_region	rc_table;
#ifdef __APPLE__
	struct _citrus_region	rc_translit_table;
#endif
	size_t			rc_src_rowcol_len;
	struct _citrus_mapper_std_linear_zone
				*rc_src_rowcol;
	_citrus_index_t		rc_src_rowcol_bits;
	_citrus_index_t		rc_src_rowcol_mask;
	_citrus_index_t		rc_dst_invalid;
	_citrus_index_t		rc_dst_unit_bits;
	int			rc_oob_mode;
	_citrus_index_t		rc_dst_ilseq;
};

struct _citrus_mapper_std;

#ifdef __APPLE__
typedef int (*_citrus_mapper_std_convert_t)(
	struct _citrus_mapper_std *__restrict,
	_index_t *__restrict, _index_t, void *__restrict,
	bool translit);
#else
typedef int (*_citrus_mapper_std_convert_t)(
	struct _citrus_mapper_std *__restrict,
	_index_t *__restrict, _index_t, void *__restrict);
#endif
typedef void (*_citrus_mapper_std_uninit_t)(struct _citrus_mapper_std *);

struct _citrus_mapper_std {
	struct _citrus_region		ms_file;
	struct _citrus_db		*ms_db;
	_citrus_mapper_std_convert_t	ms_convert;
	_citrus_mapper_std_uninit_t	ms_uninit;
	union {
		struct _citrus_mapper_std_rowcol	rowcol;
	} u;
#define ms_rowcol	u.rowcol
};

#endif
