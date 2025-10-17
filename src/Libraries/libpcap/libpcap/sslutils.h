/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 16, 2025.
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
#ifndef __SSLUTILS_H__
#define __SSLUTILS_H__

#ifdef HAVE_OPENSSL
#include "pcap/socket.h"  // for SOCKET
#include <openssl/ssl.h>
#include <openssl/err.h>

/*
 * Utility functions
 */

void ssl_set_certfile(const char *certfile);
void ssl_set_keyfile(const char *keyfile);
int ssl_init_once(int is_server, int enable_compression, char *errbuf, size_t errbuflen);
SSL *ssl_promotion(int is_server, SOCKET s, char *errbuf, size_t errbuflen);
void ssl_finish(SSL *ssl);
int ssl_send(SSL *, char const *buffer, int size, char *errbuf, size_t errbuflen);
int ssl_recv(SSL *, char *buffer, int size, char *errbuf, size_t errbuflen);

// The SSL parameters are used
#define _U_NOSSL_

#else   // HAVE_OPENSSL

// This saves us from a lot of ifdefs:
#define SSL void const

// The SSL parameters are unused
#define _U_NOSSL_	_U_

#endif  // HAVE_OPENSSL

#endif  // __SSLUTILS_H__
