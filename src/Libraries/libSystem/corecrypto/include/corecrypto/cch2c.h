/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 30, 2023.
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

#ifndef _CORECRYPTO_CCH2C_H_
#define _CORECRYPTO_CCH2C_H_

#include <corecrypto/ccec.h>
#include <stddef.h>

struct cch2c_info {
    int dummy;
};

extern const struct cch2c_info cch2c_p256_sha256_sswu_ro_info;
extern const struct cch2c_info cch2c_p384_sha512_sswu_ro_info;
extern const struct cch2c_info cch2c_p521_sha512_sswu_ro_info;

int cch2c(const struct cch2c_info *h2c_info, size_t dst_nbytes, const void *dst, size_t data_nbytes, const void *data, ccec_pub_ctx_t public);

#endif // _CORECRYPTO_CCH2C_H_