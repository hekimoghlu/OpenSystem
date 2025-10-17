/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 13, 2023.
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
/*-
 * SPDX-License-Identifier: BSD-2-Clause
 *
 * Copyright (c) 2002 Citrus Project,
 * Copyright (c) 2010 Gabor Kovesdan <gabor@FreeBSD.org>,
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

#include <sys/cdefs.h>
#include <sys/types.h>

#include <assert.h>
#include <errno.h>
#include <iconv.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#include "citrus_namespace.h"
#include "citrus_types.h"
#include "citrus_module.h"
#include "citrus_none.h"
#include "citrus_stdenc.h"

_CITRUS_STDENC_DECLS(NONE);
_CITRUS_STDENC_DEF_OPS(NONE);
struct _citrus_stdenc_traits _citrus_NONE_stdenc_traits = {
	0,	/* et_state_size */
	1,	/* mb_cur_max */
#ifdef __APPLE__
	1,	/* mb_cur_min */
#endif
};

#ifdef __APPLE__
typedef struct {
	uint32_t vmask;	/* Valid mask */
} _citrus_NONE_encoding_info;
#endif

#ifdef __APPLE__
static int
_citrus_NONE_encoding_module_init(_citrus_NONE_encoding_info * __restrict ei,
    const void * __restrict var, size_t lenvar)
{
	const char *p;

	p = var;

	while (lenvar > 0) {
		switch (_bcs_tolower(*p)) {
		case '7':
			/*
			 * For 7bit, note that one mapping may consist of up to
			 * 4 characters in the case of some transliterations;
			 * thus, the repeating pattern.
			 */
			MATCH(7bit, ei->vmask = 0x7f7f7f7f);
			break;
		}
		p++;
		lenvar--;
	}

	if (ei->vmask == 0)
		ei->vmask = ~0U;

	return (0);
}
#endif

static int
_citrus_NONE_stdenc_init(struct _citrus_stdenc * __restrict ce,
    const void *var __unused, size_t lenvar __unused,
    struct _citrus_stdenc_traits * __restrict et)
{
#ifdef __APPLE__
	_citrus_NONE_encoding_info *ei;
	int ret;

	ei = calloc(1, sizeof(*ei));
	if (ei == NULL)
		return (errno);

	ret = _citrus_NONE_encoding_module_init(ei, var, lenvar);
	if (ret != 0) {
		free(ei);
		return (ret);
	}

	ce->ce_closure = ei;

	/*
	 * Note that _citrus_NONE_stdenc_traits is effectively unused in the
	 * Apple version.  Upstream, libiconv doesn't call this encoding's
	 * stdenc_init at all.  We differ because we want to be able to specify,
	 * for example, that only 7 bits are valid.  It uses slightly more
	 * memory to do it this way, but we really need the state in case the
	 * src is a NONE encoding that can only do 7bit, but dst is a NONE
	 * encoding that can handle more.
	 */
	et->et_state_size = sizeof(_citrus_NONE_encoding_info);
	et->et_mb_cur_min = 1;
#else
	et->et_state_size = 0;
#endif
	et->et_mb_cur_max = 1;

#ifndef __APPLE__
	ce->ce_closure = NULL;
#endif

	return (0);
}

static void
#ifdef __APPLE__
_citrus_NONE_stdenc_uninit(struct _citrus_stdenc *ce)
#else
_citrus_NONE_stdenc_uninit(struct _citrus_stdenc *ce __unused)
#endif
{

#ifdef __APPLE__
	free(ce->ce_closure);
#endif
}

#ifndef __APPLE__
static int
_citrus_NONE_stdenc_init_state(struct _citrus_stdenc * __restrict ce __unused,
    void * __restrict ps __unused)
{

	return (0);
}
#endif

static int
_citrus_NONE_stdenc_mbtocs(struct _citrus_stdenc * __restrict ce __unused,
    _csid_t *csid, _index_t *idx, char **s, size_t n,
    void *ps __unused, size_t *nresult, struct iconv_hooks *hooks)
{

	if (n < 1) {
		*nresult = (size_t)-2;
		return (0);
	}

	*csid = 0;
	*idx = (_index_t)(unsigned char)*(*s)++;
	*nresult = *idx == 0 ? 0 : 1;

	if ((hooks != NULL) && (hooks->uc_hook != NULL))
		hooks->uc_hook((unsigned int)*idx, hooks->data);

	return (0);
}

static int
#ifdef __APPLE__
_citrus_NONE_stdenc_cstomb(struct _citrus_stdenc * __restrict ce,
#else
_citrus_NONE_stdenc_cstomb(struct _citrus_stdenc * __restrict ce __unused,
#endif
    char *s, size_t n, _csid_t csid, _index_t idx, void *ps __unused,
    size_t *nresult, struct iconv_hooks *hooks __unused)
{
#ifdef __APPLE__
	_citrus_NONE_encoding_info *ei;

	ei = (_citrus_NONE_encoding_info *)ce->ce_closure;
#endif

	if (csid == _CITRUS_CSID_INVALID) {
		*nresult = 0;
		return (0);
	}
	if (csid != 0)
		return (EILSEQ);

#ifdef __APPLE__
	if ((idx & ~ei->vmask) != 0)
		return (EILSEQ);
#endif
	if ((idx & 0x000000FF) == idx) {
		if (n < 1) {
			*nresult = (size_t)-1;
			return (E2BIG);
		}
		*s = (char)idx;
		*nresult = 1;
	} else if ((idx & 0x0000FFFF) == idx) {
		if (n < 2) {
			*nresult = (size_t)-1;
			return (E2BIG);
		}
		s[0] = (char)idx;
		/* XXX: might be endian dependent */
		s[1] = (char)(idx >> 8);
		*nresult = 2;
	} else if ((idx & 0x00FFFFFF) == idx) {
		if (n < 3) {
			*nresult = (size_t)-1;
			return (E2BIG);
		}
		s[0] = (char)idx;
		/* XXX: might be endian dependent */
		s[1] = (char)(idx >> 8);
		s[2] = (char)(idx >> 16);
		*nresult = 3;
	} else {
		if (n < 4) {
			*nresult = (size_t)-1;
			return (E2BIG);
		}
		s[0] = (char)idx;
		/* XXX: might be endian dependent */
		s[1] = (char)(idx >> 8);
		s[2] = (char)(idx >> 16);
		s[3] = (char)(idx >> 24);
		*nresult = 4;
	}
		
	return (0);
}

static int
_citrus_NONE_stdenc_mbtowc(struct _citrus_stdenc * __restrict ce __unused,
    _wc_t * __restrict pwc, char ** __restrict s, size_t n,
    void * __restrict pspriv __unused, size_t * __restrict nresult,
    struct iconv_hooks *hooks)
{

	if (*s == NULL) {
		*nresult = 0;
		return (0);
	}
	if (n == 0) {
		*nresult = (size_t)-2;
		return (0);
	}

	if (pwc != NULL)
		*pwc = (_wc_t)(unsigned char) **s;

	*nresult = **s == '\0' ? 0 : 1;

	if ((hooks != NULL) && (hooks->wc_hook != NULL))
		hooks->wc_hook(*pwc, hooks->data);

	return (0);
}

static int
_citrus_NONE_stdenc_wctomb(struct _citrus_stdenc * __restrict ce __unused,
    char * __restrict s, size_t n, _wc_t wc,
    void * __restrict pspriv __unused, size_t * __restrict nresult,
    struct iconv_hooks *hooks __unused)
{

	if ((wc & ~0xFFU) != 0) {
		*nresult = (size_t)-1;
		return (EILSEQ);
	}
	if (n == 0) {
		*nresult = (size_t)-1;
		return (E2BIG);
	}

	*nresult = 1;
	if (s != NULL && n > 0)
		*s = (char)wc;

	return (0);
}

static int
_citrus_NONE_stdenc_put_state_reset(struct _citrus_stdenc * __restrict ce __unused,
    char * __restrict s __unused, size_t n __unused,
    void * __restrict pspriv __unused, size_t * __restrict nresult)
{

	*nresult = 0;

	return (0);
}

static int
_citrus_NONE_stdenc_get_state_desc(struct _stdenc * __restrict ce __unused,
    void * __restrict ps __unused, int id,
    struct _stdenc_state_desc * __restrict d)
{
	int ret = 0;

	switch (id) {
	case _STDENC_SDID_GENERIC:
		d->u.generic.state = _STDENC_SDGEN_INITIAL;
		break;
	default:
		ret = EOPNOTSUPP;
	}

	return (ret);
}

