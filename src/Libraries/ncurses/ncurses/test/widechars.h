/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 16, 2025.
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
#ifndef __WIDECHARS_H
#define __WIDECHARS_H 1

#include <test.priv.h>

#if USE_WIDEC_SUPPORT

#if defined(__MINGW32__)
/*
 * MinGW has wide-character functions, but they do not work correctly.
 */

extern int _nc_mbtowc(wchar_t *pwc, const char *s, size_t n);
extern int __MINGW_NOTHROW _nc_mbtowc(wchar_t *pwc, const char *s, size_t n);
#define mbtowc(pwc,s,n) _nc_mbtowc(pwc,s,n)

extern int __MINGW_NOTHROW _nc_mblen(const char *, size_t);
#define mblen(s,n) _nc_mblen(s, n)

#endif /* __MINGW32__ */

#if HAVE_MBTOWC && HAVE_MBLEN
#define reset_mbytes(state) IGNORE_RC(mblen(NULL, 0)), IGNORE_RC(mbtowc(NULL, NULL, 0))
#define count_mbytes(buffer,length,state) mblen(buffer,length)
#define check_mbytes(wch,buffer,length,state) \
	(int) mbtowc(&wch, buffer, length)
#define state_unused
#elif HAVE_MBRTOWC && HAVE_MBRLEN
#define reset_mbytes(state) init_mb(state)
#define count_mbytes(buffer,length,state) mbrlen(buffer,length,&state)
#define check_mbytes(wch,buffer,length,state) \
	(int) mbrtowc(&wch, buffer, length, &state)
#else
make an error
#endif

#else

#endif /* USE_WIDEC_SUPPORT */

extern void widechars_stub(void);

#endif /* __WIDECHARS_H */
