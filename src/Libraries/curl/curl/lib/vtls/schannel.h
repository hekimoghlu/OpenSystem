/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 23, 2024.
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

#ifndef HEADER_CURL_SCHANNEL_H
#define HEADER_CURL_SCHANNEL_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) Marc Hoersken, <info@marc-hoersken.de>, et al.
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

#ifdef USE_SCHANNEL

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4201)
#endif
#include <subauth.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif
/* Wincrypt must be included before anything that could include OpenSSL. */
#if defined(USE_WIN32_CRYPTO)
#include <wincrypt.h>
/* Undefine wincrypt conflicting symbols for BoringSSL. */
#undef X509_NAME
#undef X509_EXTENSIONS
#undef PKCS7_ISSUER_AND_SERIAL
#undef PKCS7_SIGNER_INFO
#undef OCSP_REQUEST
#undef OCSP_RESPONSE
#endif

#include <schnlsp.h>
#include <schannel.h>
#include "curl_sspi.h"

#include "cfilters.h"
#include "urldata.h"

/* <wincrypt.h> has been included via the above <schnlsp.h>.
 * Or in case of ldap.c, it was included via <winldap.h>.
 * And since <wincrypt.h> has this:
 *   #define X509_NAME  ((LPCSTR) 7)
 *
 * And in BoringSSL's <openssl/base.h> there is:
 *  typedef struct X509_name_st X509_NAME;
 *  etc.
 *
 * this will cause all kinds of C-preprocessing paste errors in
 * BoringSSL's <openssl/x509.h>: So just undefine those defines here
 * (and only here).
 */
#if defined(OPENSSL_IS_BORINGSSL)
# undef X509_NAME
# undef X509_CERT_PAIR
# undef X509_EXTENSIONS
#endif

extern const struct Curl_ssl Curl_ssl_schannel;

CURLcode Curl_verify_host(struct Curl_cfilter *cf,
                          struct Curl_easy *data);

CURLcode Curl_verify_certificate(struct Curl_cfilter *cf,
                                 struct Curl_easy *data);

#endif /* USE_SCHANNEL */
#endif /* HEADER_CURL_SCHANNEL_H */
