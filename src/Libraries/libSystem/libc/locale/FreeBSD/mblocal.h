/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 19, 2022.
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

/*
 * Rune initialization function prototypes.
 */
__attribute__((visibility("hidden"))) int	_none_init(struct __xlocale_st_runelocale *);
__attribute__((visibility("hidden"))) int	_ascii_init(struct __xlocale_st_runelocale *);
__attribute__((visibility("hidden"))) int	_UTF2_init(struct __xlocale_st_runelocale *);
__attribute__((visibility("hidden"))) int	_UTF8_init(struct __xlocale_st_runelocale *);
__attribute__((visibility("hidden"))) int	_EUC_init(struct __xlocale_st_runelocale *);
__attribute__((visibility("hidden"))) int	_GB18030_init(struct __xlocale_st_runelocale *);
__attribute__((visibility("hidden"))) int	_GB2312_init(struct __xlocale_st_runelocale *);
__attribute__((visibility("hidden"))) int	_GBK_init(struct __xlocale_st_runelocale *);
__attribute__((visibility("hidden"))) int	_BIG5_init(struct __xlocale_st_runelocale *);
__attribute__((visibility("hidden"))) int	_MSKanji_init(struct __xlocale_st_runelocale *);

__attribute__((visibility("hidden"))) size_t       _none_mbrtowc(wchar_t * __restrict, const char * __restrict,
                    size_t, mbstate_t * __restrict, locale_t);
__attribute__((visibility("hidden"))) int  _none_mbsinit(const mbstate_t *, locale_t);
__attribute__((visibility("hidden"))) size_t       _none_mbsnrtowcs(wchar_t * __restrict dst,
                    const char ** __restrict src, size_t nms, size_t len,
                    mbstate_t * __restrict ps __unused, locale_t);
__attribute__((visibility("hidden"))) size_t       _none_wcrtomb(char * __restrict, wchar_t,
                    mbstate_t * __restrict, locale_t);
__attribute__((visibility("hidden"))) size_t       _none_wcsnrtombs(char * __restrict, const wchar_t ** __restrict,
                    size_t, size_t, mbstate_t * __restrict, locale_t);

extern size_t __mbsnrtowcs_std(wchar_t * __restrict, const char ** __restrict,
    size_t, size_t, mbstate_t * __restrict, locale_t);
extern size_t __wcsnrtombs_std(char * __restrict, const wchar_t ** __restrict,
    size_t, size_t, mbstate_t * __restrict, locale_t);

#endif	/* _MBLOCAL_H_ */
