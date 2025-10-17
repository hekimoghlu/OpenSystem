/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 8, 2024.
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

#ifndef HEADER_CURL_HSTS_H
#define HEADER_CURL_HSTS_H
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
#include "curl_setup.h"

#if !defined(CURL_DISABLE_HTTP) && !defined(CURL_DISABLE_HSTS)
#include <curl/curl.h>
#include "llist.h"

#ifdef DEBUGBUILD
extern time_t deltatime;
#endif

struct stsentry {
  struct Curl_llist_element node;
  const char *host;
  bool includeSubDomains;
  curl_off_t expires; /* the timestamp of this entry's expiry */
};

/* The HSTS cache. Needs to be able to tailmatch host names. */
struct hsts {
  struct Curl_llist list;
  char *filename;
  unsigned int flags;
};

struct hsts *Curl_hsts_init(void);
void Curl_hsts_cleanup(struct hsts **hp);
CURLcode Curl_hsts_parse(struct hsts *h, const char *hostname,
                         const char *sts);
struct stsentry *Curl_hsts(struct hsts *h, const char *hostname,
                           bool subdomain);
CURLcode Curl_hsts_save(struct Curl_easy *data, struct hsts *h,
                        const char *file);
CURLcode Curl_hsts_loadfile(struct Curl_easy *data,
                            struct hsts *h, const char *file);
CURLcode Curl_hsts_loadcb(struct Curl_easy *data,
                          struct hsts *h);
void Curl_hsts_loadfiles(struct Curl_easy *data);
#else
#define Curl_hsts_cleanup(x)
#define Curl_hsts_loadcb(x,y) CURLE_OK
#define Curl_hsts_save(x,y,z)
#define Curl_hsts_loadfiles(x)
#endif /* CURL_DISABLE_HTTP || CURL_DISABLE_HSTS */
#endif /* HEADER_CURL_HSTS_H */
