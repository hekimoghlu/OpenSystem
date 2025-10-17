/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 11, 2023.
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
/* $NetBSD: citrus_stdenc_local.h,v 1.4 2008/02/09 14:56:20 junyoung Exp $ */

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
 *
 */

#ifndef _CITRUS_STDENC_LOCAL_H_
#define _CITRUS_STDENC_LOCAL_H_

#include <iconv.h>

#include "citrus_module.h"

#ifdef __APPLE__
#ifndef _ENCODING_HAVE_MBTOCSN
#define _ENCODING_HAVE_MBTOCSN 0
#endif

#ifndef _ENCODING_NEED_INIT_STATE
#define	_ENCODING_NEED_INIT_STATE	0
#endif

typedef void (_citrus_save_encoding_state_t)(void *);
#else /* !__APPLE__ */
#define _ENCODING_HAVE_MBTOCSN 0
#define	_ENCODING_NEED_INIT_STATE	1
#endif /* __APPLE__ */

#define _CITRUS_STDENC_GETOPS_FUNC_BASE(n)			\
   int n(struct _citrus_stdenc_ops *, size_t)
#define _CITRUS_STDENC_GETOPS_FUNC(_e_)					\
   _CITRUS_STDENC_GETOPS_FUNC_BASE(_citrus_##_e_##_stdenc_getops)
typedef _CITRUS_STDENC_GETOPS_FUNC_BASE((*_citrus_stdenc_getops_t));

#if _ENCODING_HAVE_MBTOCSN
#define	_ENCODING_STDENC_MBTOCSN(_e_)	\
    &_citrus_##_e_##_stdenc_mbtocsn
#define	_CITRUS_STDENC_DECL_MBTOCSN(_e_)				\
static int	 _citrus_##_e_##_stdenc_mbtocsn				\
		    (struct _citrus_stdenc * __restrict,		\
		    _citrus_csid_t * __restrict,			\
		    _citrus_index_t * __restrict,			\
		    unsigned short * __restrict,			\
		    int * __restrict,					\
		    char ** __restrict, size_t,				\
		    void * __restrict, size_t * __restrict,		\
		    struct iconv_hooks *,				\
		    _citrus_save_encoding_state_t *, void *)

#define	_ENCODING_STDENC_CSTOMBN(_e_)	\
    &_citrus_##_e_##_stdenc_cstombn
#define	_CITRUS_STDENC_DECL_CSTOMBN(_e_)				\
static int	 _citrus_##_e_##_stdenc_cstombn				\
		    (struct _citrus_stdenc * __restrict,		\
		    char * __restrict, size_t,				\
		    _citrus_csid_t * __restrict,			\
		    _citrus_index_t * __restrict,			\
		    int * __restrict cnt,				\
		    void * __restrict, size_t * __restrict,		\
		    struct iconv_hooks *,				\
		    _citrus_save_encoding_state_t *, void *)
#else
#define _ENCODING_STDENC_MBTOCSN(_e_)		NULL
#define _CITRUS_STDENC_DECL_MBTOCSN(_e_)

#define _ENCODING_STDENC_CSTOMBN(_e_)		NULL
#define	_CITRUS_STDENC_DECL_CSTOMBN(_e_)
#endif

#if _ENCODING_NEED_INIT_STATE
#define	_ENCODING_STDENC_INIT_STATE(_e_)	\
    &_citrus_##_e_##_stdenc_init_state
#define	_CITRUS_STDENC_DECL_INIT_STATE(_e_)				\
static int	 _citrus_##_e_##_stdenc_init_state			\
		    (struct _citrus_stdenc * __restrict,		\
		    void * __restrict)
#else
#define _ENCODING_STDENC_INIT_STATE(_e_)	NULL
#define	_CITRUS_STDENC_DECL_INIT_STATE(_e_)
#endif

#define _CITRUS_STDENC_DECLS(_e_)					\
static int	 _citrus_##_e_##_stdenc_init				\
		    (struct _citrus_stdenc * __restrict,		\
		    const void * __restrict, size_t,			\
		    struct _citrus_stdenc_traits * __restrict);		\
static void	 _citrus_##_e_##_stdenc_uninit(struct _citrus_stdenc *);\
_CITRUS_STDENC_DECL_INIT_STATE(_e_);					\
static int	 _citrus_##_e_##_stdenc_mbtocs				\
		    (struct _citrus_stdenc * __restrict,		\
		    _citrus_csid_t * __restrict,			\
		    _citrus_index_t * __restrict,			\
		    char ** __restrict, size_t,				\
		    void * __restrict, size_t * __restrict,		\
		    struct iconv_hooks *);				\
_CITRUS_STDENC_DECL_MBTOCSN(_e_);					\
static int	 _citrus_##_e_##_stdenc_cstomb				\
		    (struct _citrus_stdenc * __restrict,		\
		    char * __restrict, size_t, _citrus_csid_t,		\
		    _citrus_index_t, void * __restrict,			\
		    size_t * __restrict, struct iconv_hooks *);		\
_CITRUS_STDENC_DECL_CSTOMBN(_e_);					\
static int	 _citrus_##_e_##_stdenc_mbtowc				\
		    (struct _citrus_stdenc * __restrict,		\
		    _citrus_wc_t * __restrict,				\
		    char ** __restrict, size_t,				\
		    void * __restrict, size_t * __restrict,		\
		    struct iconv_hooks *);				\
static int	 _citrus_##_e_##_stdenc_wctomb				\
		    (struct _citrus_stdenc * __restrict,		\
		    char * __restrict, size_t, _citrus_wc_t,		\
		    void * __restrict, size_t * __restrict,		\
		    struct iconv_hooks *);				\
static int	 _citrus_##_e_##_stdenc_put_state_reset			\
		    (struct _citrus_stdenc * __restrict,		\
		    char * __restrict, size_t, void * __restrict,	\
		    size_t * __restrict);				\
static int	 _citrus_##_e_##_stdenc_get_state_desc			\
		    (struct _citrus_stdenc * __restrict,		\
		    void * __restrict, int,				\
		    struct _citrus_stdenc_state_desc * __restrict)

#define _CITRUS_STDENC_DEF_OPS(_e_)					\
extern struct _citrus_stdenc_ops _citrus_##_e_##_stdenc_ops;		\
struct _citrus_stdenc_ops _citrus_##_e_##_stdenc_ops = {		\
	/* eo_init */		&_citrus_##_e_##_stdenc_init,		\
	/* eo_uninit */		&_citrus_##_e_##_stdenc_uninit,		\
	/* eo_init_state */	_ENCODING_STDENC_INIT_STATE(_e_),	\
	/* eo_mbtocs */		&_citrus_##_e_##_stdenc_mbtocs,		\
	/* eo_cstomb */		&_citrus_##_e_##_stdenc_cstomb,		\
	/* eo_mbtowc */		&_citrus_##_e_##_stdenc_mbtowc,		\
	/* eo_wctomb */		&_citrus_##_e_##_stdenc_wctomb,		\
	/* eo_put_state_reset */&_citrus_##_e_##_stdenc_put_state_reset,\
	/* eo_get_state_desc */	&_citrus_##_e_##_stdenc_get_state_desc,	\
	/* eo_mbtocsn */	_ENCODING_STDENC_MBTOCSN(_e_),		\
	/* eo_cstombn */	_ENCODING_STDENC_CSTOMBN(_e_),		\
}

typedef int (*_citrus_stdenc_init_t)
    (struct _citrus_stdenc * __reatrict, const void * __restrict , size_t,
    struct _citrus_stdenc_traits * __restrict);
typedef void (*_citrus_stdenc_uninit_t)(struct _citrus_stdenc * __restrict);
typedef int (*_citrus_stdenc_init_state_t)
    (struct _citrus_stdenc * __restrict, void * __restrict);
typedef int (*_citrus_stdenc_mbtocs_t)
    (struct _citrus_stdenc * __restrict,
    _citrus_csid_t * __restrict, _citrus_index_t * __restrict,
    char ** __restrict, size_t,
    void * __restrict, size_t * __restrict,
    struct iconv_hooks *);
typedef int (*_citrus_stdenc_cstomb_t)
    (struct _citrus_stdenc *__restrict, char * __restrict, size_t,
    _citrus_csid_t, _citrus_index_t, void * __restrict,
    size_t * __restrict, struct iconv_hooks *);
#ifdef __APPLE__
typedef int (*_citrus_stdenc_mbtocsn_t)
    (struct _citrus_stdenc * __restrict,
    _citrus_csid_t * __restrict, _citrus_index_t * __restrict,
    unsigned short * __restrict, int * __restrict,
    char ** __restrict, size_t,
    void * __restrict, size_t * __restrict,
    struct iconv_hooks *,
    _citrus_save_encoding_state_t *save_state, void *save_state_cookie);
typedef int (*_citrus_stdenc_cstombn_t)
    (struct _citrus_stdenc *__restrict, char * __restrict, size_t,
    _citrus_csid_t * __restrict, _citrus_index_t * __restrict,
    int * __restrict, void * __restrict, size_t * __restrict,
    struct iconv_hooks *,
    _citrus_save_encoding_state_t *save_state, void *save_state_cookie);
#endif
typedef int (*_citrus_stdenc_mbtowc_t)
    (struct _citrus_stdenc * __restrict,
    _citrus_wc_t * __restrict,
    char ** __restrict, size_t,
    void * __restrict, size_t * __restrict,
    struct iconv_hooks *);
typedef int (*_citrus_stdenc_wctomb_t)
    (struct _citrus_stdenc *__restrict, char * __restrict, size_t,
    _citrus_wc_t, void * __restrict, size_t * __restrict,
    struct iconv_hooks *);
typedef int (*_citrus_stdenc_put_state_reset_t)
    (struct _citrus_stdenc *__restrict, char * __restrict, size_t,
    void * __restrict, size_t * __restrict);
typedef int (*_citrus_stdenc_get_state_desc_t)
    (struct _citrus_stdenc * __restrict, void * __restrict, int,
    struct _citrus_stdenc_state_desc * __restrict);

struct _citrus_stdenc_ops {
	_citrus_stdenc_init_t		eo_init;
	_citrus_stdenc_uninit_t		eo_uninit;
	_citrus_stdenc_init_state_t	eo_init_state;
	_citrus_stdenc_mbtocs_t		eo_mbtocs;
	_citrus_stdenc_cstomb_t		eo_cstomb;
	_citrus_stdenc_mbtowc_t		eo_mbtowc;
	_citrus_stdenc_wctomb_t		eo_wctomb;
	_citrus_stdenc_put_state_reset_t eo_put_state_reset;
	/* version 0x00000002 */
	_citrus_stdenc_get_state_desc_t	eo_get_state_desc;
#ifdef __APPLE__
	/* version ... */
	_citrus_stdenc_mbtocsn_t	eo_mbtocsn;
	_citrus_stdenc_cstombn_t	eo_cstombn;
#endif
};

struct _citrus_stdenc_traits {
	/* version 0x00000001 */
	size_t				 et_state_size;
	size_t				 et_mb_cur_max;
	/* version 0x00000005 */
	size_t				 et_mb_cur_min;
};

struct _citrus_stdenc {
	/* version 0x00000001 */
	struct _citrus_stdenc_ops	*ce_ops;
	void				*ce_closure;
	_citrus_module_t		 ce_module;
	struct _citrus_stdenc_traits	*ce_traits;
};

#define _CITRUS_DEFAULT_STDENC_NAME		"NONE"

#endif
