/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 8, 2022.
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

#ifndef HEADER_CURL_BASE64_H
#define HEADER_CURL_BASE64_H
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

#ifndef BUILDING_LIBCURL
/* this renames functions so that the tool code can use the same code
   without getting symbol collisions */
#define Curl_base64_encode(a,b,c,d) curlx_base64_encode(a,b,c,d)
#define Curl_base64url_encode(a,b,c,d) curlx_base64url_encode(a,b,c,d)
#define Curl_base64_decode(a,b,c) curlx_base64_decode(a,b,c)
#endif

CURLcode Curl_base64_encode(const char *inputbuff, size_t insize,
                            char **outptr, size_t *outlen);
CURLcode Curl_base64url_encode(const char *inputbuff, size_t insize,
                               char **outptr, size_t *outlen);
CURLcode Curl_base64_decode(const char *src,
                            unsigned char **outptr, size_t *outlen);
#endif /* HEADER_CURL_BASE64_H */
