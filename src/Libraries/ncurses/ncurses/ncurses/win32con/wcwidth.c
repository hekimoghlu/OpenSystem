/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 20, 2022.
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
#include <curses.priv.h>

MODULE_ID("$Id: wcwidth.c,v 1.2 2010/08/07 20:52:43 tom Exp $")

#if USE_WIDEC_SUPPORT
#define mk_wcwidth(ucs)          _nc_wcwidth(ucs)
#define mk_wcswidth(pwcs, n)     _nc_wcswidth(pwcs, n)
#define mk_wcwidth_cjk(ucs)      _nc_wcwidth_cjk(ucs)
#define mk_wcswidth_cjk(pwcs, n) _nc_wcswidth_cjk(pwcs, n)

extern int mk_wcwidth(wchar_t);
extern int mk_wcswidth(const wchar_t *, size_t);
extern int mk_wcwidth_cjk(wchar_t);
extern int mk_wcswidth_cjk(const wchar_t *, size_t);

#include <wcwidth.h>
#else
void _nc_empty_wcwidth(void);
void
_nc_empty_wcwidth(void)
{
}
#endif
