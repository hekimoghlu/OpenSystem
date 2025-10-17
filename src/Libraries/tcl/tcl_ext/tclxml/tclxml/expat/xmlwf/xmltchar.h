/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 15, 2025.
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

#ifdef XML_UNICODE
#ifndef XML_UNICODE_WCHAR_T
#error xmlwf requires a 16-bit Unicode-compatible wchar_t 
#endif
#define T(x) L ## x
#define ftprintf fwprintf
#define tfopen _wfopen
#define fputts fputws
#define puttc putwc
#define tcscmp wcscmp
#define tcscpy wcscpy
#define tcscat wcscat
#define tcschr wcschr
#define tcsrchr wcsrchr
#define tcslen wcslen
#define tperror _wperror
#define topen _wopen
#define tmain wmain
#define tremove _wremove
#else /* not XML_UNICODE */
#define T(x) x
#define ftprintf fprintf
#define tfopen fopen
#define fputts fputs
#define puttc putc
#define tcscmp strcmp
#define tcscpy strcpy
#define tcscat strcat
#define tcschr strchr
#define tcsrchr strrchr
#define tcslen strlen
#define tperror perror
#define topen open
#define tmain main
#define tremove remove
#endif /* not XML_UNICODE */
