/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 1, 2024.
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

#ifndef HEADER_CURL_KRB5_H
#define HEADER_CURL_KRB5_H
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

struct Curl_sec_client_mech {
  const char *name;
  size_t size;
  int (*init)(void *);
  int (*auth)(void *, struct Curl_easy *data, struct connectdata *);
  void (*end)(void *);
  int (*check_prot)(void *, int);
  int (*encode)(void *, const void *, int, int, void **);
  int (*decode)(void *, void *, int, int, struct connectdata *);
};

#define AUTH_OK         0
#define AUTH_CONTINUE   1
#define AUTH_ERROR      2

#ifdef HAVE_GSSAPI
int Curl_sec_read_msg(struct Curl_easy *data, struct connectdata *conn, char *,
                      enum protection_level);
void Curl_sec_end(struct connectdata *);
CURLcode Curl_sec_login(struct Curl_easy *, struct connectdata *);
int Curl_sec_request_prot(struct connectdata *conn, const char *level);
#else
#define Curl_sec_end(x)
#endif

#endif /* HEADER_CURL_KRB5_H */
