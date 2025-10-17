/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 28, 2024.
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
#ifndef OPENSSL_HEADER_LHASH_H
#define OPENSSL_HEADER_LHASH_H

#include "CNIOBoringSSL_base.h"

#if defined(__cplusplus)
extern "C" {
#endif


// lhash is an internal library and not exported for use outside BoringSSL. This
// header is provided for compatibility with code that expects OpenSSL.


// These two macros are exported for compatibility with existing callers of
// |X509V3_EXT_conf_nid|. Do not use these symbols outside BoringSSL.
#define LHASH_OF(type) struct lhash_st_##type
#define DECLARE_LHASH_OF(type) LHASH_OF(type);


#if defined(__cplusplus)
}  // extern C
#endif

#endif  // OPENSSL_HEADER_LHASH_H
