/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 6, 2025.
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
#ifndef _MBLOCAL_H_
#define	_MBLOCAL_H_

#include <runetype.h>
#include "xlocale_private.h"

#define	SS2	0x008e
#define SS3	0x008f

struct xlocale_ctype {
	struct xlocale_component header;
	int __mb_cur_max;
	int __mb_sb_limit;
	size_t (*__mbrtowc)(wchar_t * __restrict, const char * __restrict,
	    size_t, __darwin_mbstate_t * __restrict, struct _xlocale *);
	int (*__mbsinit)(const __darwin_mbstate_t *, struct _xlocale *);
	size_t (*__mbsnrtowcs)(wchar_t * __restrict, const char ** __restrict,
	    size_t, size_t, __darwin_mbstate_t * __restrict, struct _xlocale *);
	size_t (*__wcrtomb)(char * __restrict, wchar_t,
	    __darwin_mbstate_t * __restrict, struct _xlocale *);
	size_t (*__wcsnrtombs)(char * __restrict, const wchar_t ** __restrict,
	    size_t, size_t, __darwin_mbstate_t * __restrict, struct _xlocale *);
	int __datasize;
	_RuneLocale *_CurrentRuneLocale;
};

/*
 * Rune initialization function prototypes.
 */
__attribute__((visibility("hidden"))) int	_none_init(struct xlocale_ctype *);
__attribute__((visibility("hidden"))) int	_ascii_init(struct xlocale_ctype *);
__attribute__((visibility("hidden"))) int	_UTF2_init(struct xlocale_ctype *);
__attribute__((visibility("hidden"))) int	_UTF8_init(struct xlocale_ctype *);
__attribute__((visibility("hidden"))) int	_EUC_CN_init(struct xlocale_ctype *);
__attribute__((visibility("hidden"))) int	_EUC_JP_init(struct xlocale_ctype *);
__attribute__((visibility("hidden"))) int	_EUC_KR_init(struct xlocale_ctype *);
__attribute__((visibility("hidden"))) int	_EUC_TW_init(struct xlocale_ctype *);
__attribute__((visibility("hidden"))) int	_EUC_init(struct xlocale_ctype *);
__attribute__((visibility("hidden"))) int	_GB18030_init(struct xlocale_ctype *);
__attribute__((visibility("hidden"))) int	_GB2312_init(struct xlocale_ctype *);
__attribute__((visibility("hidden"))) int	_GBK_init(struct xlocale_ctype *);
__attribute__((visibility("hidden"))) int	_BIG5_init(struct xlocale_ctype *);
__attribute__((visibility("hidden"))) int	_MSKanji_init(struct xlocale_ctype *);

__attribute__((visibility("hidden"))) size_t       _none_mbrtowc(wchar_t * __restrict, const char * __restrict,
                    size_t, __darwin_mbstate_t * __restrict, locale_t);
__attribute__((visibility("hidden"))) int  _none_mbsinit(const __darwin_mbstate_t *, locale_t);
__attribute__((visibility("hidden"))) size_t       _none_mbsnrtowcs(wchar_t * __restrict dst,
                    const char ** __restrict src, size_t nms, size_t len,
                    __darwin_mbstate_t * __restrict ps __unused, locale_t);
__attribute__((visibility("hidden"))) size_t       _none_wcrtomb(char * __restrict, wchar_t,
                    __darwin_mbstate_t * __restrict, locale_t);
__attribute__((visibility("hidden"))) size_t       _none_wcsnrtombs(char * __restrict, const wchar_t ** __restrict,
                    size_t, size_t, __darwin_mbstate_t * __restrict, locale_t);

extern size_t __mbsnrtowcs_std(wchar_t * __restrict, const char ** __restrict,
    size_t, size_t, __darwin_mbstate_t * __restrict, locale_t);
extern size_t __wcsnrtombs_std(char * __restrict, const wchar_t ** __restrict,
    size_t, size_t, __darwin_mbstate_t * __restrict, locale_t);

#endif	/* _MBLOCAL_H_ */
