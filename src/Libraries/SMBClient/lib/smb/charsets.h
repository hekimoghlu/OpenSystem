/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 4, 2021.
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
#if !defined(__CHARSETS_H__)
#define __CHARSETS_H__ 1

#include <CoreFoundation/CoreFoundation.h>

void str_upper(char *dst, size_t maxDstLen, CFStringRef srcRef);
extern char *convert_wincs_to_utf8(const char *windows_string, CFStringEncoding codePage);
extern char *convert_utf8_to_wincs(const char *utf8_string, CFStringEncoding codePage, int uppercase);
extern char *convert_leunicode_to_utf8(unsigned short *windows_string, size_t maxLen);
extern char *convert_unicode_to_utf8(const uint16_t *unicode_string, size_t maxLen, uint32_t decompose);
extern unsigned short *convert_utf8_to_leunicode(const char *utf8_string, size_t utf8_maxLen);
#endif /* !__CHARSETS_H__ */
