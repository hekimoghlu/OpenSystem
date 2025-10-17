/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 9, 2025.
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

#ifndef HEADER_CURL_CTYPE_H
#define HEADER_CURL_CTYPE_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) Daniel Stenberg, <daniel@haxx.se>, et al.
 *
 * This software is licensed as described in the file COPYING, which
 * you should have received as part of this distribution. The terms
 * are also available at https://curl.se/docs/copyright.html.
 *
 * You may opt to use, copy, modify, merge, publish, distribute and/or sell
 * copies of the Software, and permit persons to whom the Software is
 * furnished to do so, under the terms of the COPYING file.
 *
 * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
 * KIND, either express or implied.
 *
 * SPDX-License-Identifier: curl
 *
 ***************************************************************************/

#define ISLOWHEXALHA(x) (((x) >= 'a') && ((x) <= 'f'))
#define ISUPHEXALHA(x) (((x) >= 'A') && ((x) <= 'F'))

#define ISLOWCNTRL(x) ((unsigned char)(x) <= 0x1f)
#define IS7F(x) ((x) == 0x7f)

#define ISLOWPRINT(x) (((x) >= 9) && ((x) <= 0x0d))

#define ISPRINT(x)  (ISLOWPRINT(x) || (((x) >= ' ') && ((x) <= 0x7e)))
#define ISGRAPH(x)  (ISLOWPRINT(x) || (((x) > ' ') && ((x) <= 0x7e)))
#define ISCNTRL(x) (ISLOWCNTRL(x) || IS7F(x))
#define ISALPHA(x) (ISLOWER(x) || ISUPPER(x))
#define ISXDIGIT(x) (ISDIGIT(x) || ISLOWHEXALHA(x) || ISUPHEXALHA(x))
#define ISALNUM(x)  (ISDIGIT(x) || ISLOWER(x) || ISUPPER(x))
#define ISUPPER(x)  (((x) >= 'A') && ((x) <= 'Z'))
#define ISLOWER(x)  (((x) >= 'a') && ((x) <= 'z'))
#define ISDIGIT(x)  (((x) >= '0') && ((x) <= '9'))
#define ISBLANK(x)  (((x) == ' ') || ((x) == '\t'))
#define ISSPACE(x)  (ISBLANK(x) || (((x) >= 0xa) && ((x) <= 0x0d)))
#define ISURLPUNTCS(x) (((x) == '-') || ((x) == '.') || ((x) == '_') || \
                        ((x) == '~'))
#define ISUNRESERVED(x) (ISALNUM(x) || ISURLPUNTCS(x))


#endif /* HEADER_CURL_CTYPE_H */
