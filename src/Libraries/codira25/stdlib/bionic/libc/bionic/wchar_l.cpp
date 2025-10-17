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
#include <wchar.h>
#include <xlocale.h>

int wcscasecmp_l(const wchar_t* ws1, const wchar_t* ws2, locale_t) {
  return wcscasecmp(ws1, ws2);
}

int wcsncasecmp_l(const wchar_t* ws1, const wchar_t* ws2, size_t n, locale_t) {
  return wcsncasecmp(ws1, ws2, n);
}

int wcscoll_l(const wchar_t* ws1, const wchar_t* ws2, locale_t) {
  return wcscoll(ws1, ws2);
}

size_t wcsxfrm_l(wchar_t* dst, const wchar_t* src, size_t n, locale_t) {
  return wcsxfrm(dst, src, n);
}
