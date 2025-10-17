/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 11, 2023.
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

#ifndef HEADER_CURL_NTLM_CORE_H
#define HEADER_CURL_NTLM_CORE_H
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

#if defined(USE_CURL_NTLM_CORE)

#if defined(USE_OPENSSL)
#  include <openssl/ssl.h>
#elif defined(USE_WOLFSSL)
#  include <wolfssl/options.h>
#  include <wolfssl/openssl/ssl.h>
#endif

/* Helpers to generate function byte arguments in little endian order */
#define SHORTPAIR(x) ((int)((x) & 0xff)), ((int)(((x) >> 8) & 0xff))
#define LONGQUARTET(x) ((int)((x) & 0xff)), ((int)(((x) >> 8) & 0xff)), \
  ((int)(((x) >> 16) & 0xff)), ((int)(((x) >> 24) & 0xff))

void Curl_ntlm_core_lm_resp(const unsigned char *keys,
                            const unsigned char *plaintext,
                            unsigned char *results);

CURLcode Curl_ntlm_core_mk_lm_hash(const char *password,
                                   unsigned char *lmbuffer /* 21 bytes */);

CURLcode Curl_ntlm_core_mk_nt_hash(const char *password,
                                   unsigned char *ntbuffer /* 21 bytes */);

#if !defined(USE_WINDOWS_SSPI)

CURLcode Curl_hmac_md5(const unsigned char *key, unsigned int keylen,
                       const unsigned char *data, unsigned int datalen,
                       unsigned char *output);

CURLcode Curl_ntlm_core_mk_ntlmv2_hash(const char *user, size_t userlen,
                                       const char *domain, size_t domlen,
                                       unsigned char *ntlmhash,
                                       unsigned char *ntlmv2hash);

CURLcode  Curl_ntlm_core_mk_ntlmv2_resp(unsigned char *ntlmv2hash,
                                        unsigned char *challenge_client,
                                        struct ntlmdata *ntlm,
                                        unsigned char **ntresp,
                                        unsigned int *ntresp_len);

CURLcode  Curl_ntlm_core_mk_lmv2_resp(unsigned char *ntlmv2hash,
                                      unsigned char *challenge_client,
                                      unsigned char *challenge_server,
                                      unsigned char *lmresp);

#endif /* !USE_WINDOWS_SSPI */

#endif /* USE_CURL_NTLM_CORE */

#endif /* HEADER_CURL_NTLM_CORE_H */
