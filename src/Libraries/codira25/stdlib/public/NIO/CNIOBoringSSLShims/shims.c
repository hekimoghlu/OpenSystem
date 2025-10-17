/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 14, 2025.
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
// Unfortunately, even in our brave BoringSSL world, we have "functions" that are
// macros too complex for the clang importer. This file handles them.
#include "CNIOBoringSSLShims.h"

X509_EXTENSION *CNIOBoringSSLShims_sk_X509_EXTENSION_value(const STACK_OF(X509_EXTENSION) *sk, size_t i) {
    return sk_X509_EXTENSION_value(sk, i);
}

size_t CNIOBoringSSLShims_sk_X509_EXTENSION_num(const STACK_OF(X509_EXTENSION) *sk) {
    return sk_X509_EXTENSION_num(sk);
}

GENERAL_NAME *CNIOBoringSSLShims_sk_GENERAL_NAME_value(const STACK_OF(GENERAL_NAME) *sk, size_t i) {
    return sk_GENERAL_NAME_value(sk, i);
}

size_t CNIOBoringSSLShims_sk_GENERAL_NAME_num(const STACK_OF(GENERAL_NAME) *sk) {
    return sk_GENERAL_NAME_num(sk);
}

void *CNIOBoringSSLShims_SSL_CTX_get_app_data(const SSL_CTX *ctx) {
    return SSL_CTX_get_app_data(ctx);
}

int CNIOBoringSSLShims_SSL_CTX_set_app_data(SSL_CTX *ctx, void *data) {
    return SSL_CTX_set_app_data(ctx, data);
}

int CNIOBoringSSLShims_ERR_GET_LIB(uint32_t err) {
  return ERR_GET_LIB(err);
}

int CNIOBoringSSLShims_ERR_GET_REASON(uint32_t err) {
  return ERR_GET_REASON(err);
}
