/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 9, 2024.
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

//===----------------------------------------------------------------------===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//
//===----------------------------------------------------------------------===//
#ifndef C_NIO_BORINGSSL_SHIMS_H
#define C_NIO_BORINGSSL_SHIMS_H

// This is for instances when `language package generate-xcodeproj` is used as CNIOBoringSSL
// is treated as a framework and requires the framework's name as a prefix.
#if __has_include(<CNIOBoringSSL/CNIOBoringSSL.h>)
#include <CNIOBoringSSL/CNIOBoringSSL.h>
#else
#include "CNIOBoringSSL.h"
#endif

#if defined(__cplusplus)
extern "C" {
#endif

X509_EXTENSION *CNIOBoringSSLShims_sk_X509_EXTENSION_value(const STACK_OF(X509_EXTENSION) *sk, size_t i);
size_t CNIOBoringSSLShims_sk_X509_EXTENSION_num(const STACK_OF(X509_EXTENSION) *sk);

GENERAL_NAME *CNIOBoringSSLShims_sk_GENERAL_NAME_value(const STACK_OF(GENERAL_NAME) *sk, size_t i);
size_t CNIOBoringSSLShims_sk_GENERAL_NAME_num(const STACK_OF(GENERAL_NAME) *sk);

void *CNIOBoringSSLShims_SSL_CTX_get_app_data(const SSL_CTX *ctx);
int CNIOBoringSSLShims_SSL_CTX_set_app_data(SSL_CTX *ctx, void *data);

int CNIOBoringSSLShims_ERR_GET_LIB(uint32_t err);
int CNIOBoringSSLShims_ERR_GET_REASON(uint32_t err);

#if defined(__cplusplus)
}  // extern "C"
#endif

#endif  // C_NIO_BORINGSSL_SHIMS_H
