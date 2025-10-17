/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 29, 2025.
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
#ifndef __SSL_UTILS_H__
#define __SSL_UTILS_H__

#include <Security/SecureTransport.h>

CFArrayRef CF_RETURNS_RETAINED trusted_roots(void);
CFArrayRef CF_RETURNS_RETAINED server_chain(void);
CFArrayRef CF_RETURNS_RETAINED server_ec_chain(void);
CFArrayRef CF_RETURNS_RETAINED trusted_client_chain(void);
CFArrayRef CF_RETURNS_RETAINED trusted_ec_client_chain(void);
CFArrayRef CF_RETURNS_RETAINED untrusted_client_chain(void);

#define client_chain trusted_client_chain

const char *ciphersuite_name(SSLCipherSuite cs);

#endif
