/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 18, 2024.
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

#ifndef _WCHAR_H_
#define _WCHAR_H_

typedef union {
       char            __mbstate8[128];
       long long       _mbstateL;                      /* for alignment */
} __mbstate_t;

typedef __mbstate_t mbstate_t;
wchar_t        *wcschr(const wchar_t *, wchar_t);
wchar_t        *wcspbrk(const wchar_t *, const wchar_t *);
wchar_t        *wcsrchr(const wchar_t *, wchar_t);
wchar_t        *wcsstr(const wchar_t * __restrict, const wchar_t * __restrict);
wchar_t        *wmemchr(const wchar_t *, wchar_t, size_t);

#endif /* !_WCHAR_H_ */
