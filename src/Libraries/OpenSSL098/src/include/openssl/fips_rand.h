/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 28, 2024.
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
#ifndef HEADER_FIPS_RAND_H
#define HEADER_FIPS_RAND_H

#include "des.h"

#ifdef OPENSSL_FIPS

#ifdef  __cplusplus
extern "C" {
#endif

int FIPS_rand_set_key(const unsigned char *key, FIPS_RAND_SIZE_T keylen);
int FIPS_rand_seed(const void *buf, FIPS_RAND_SIZE_T num);
int FIPS_rand_bytes(unsigned char *out, FIPS_RAND_SIZE_T outlen);

int FIPS_rand_test_mode(void);
void FIPS_rand_reset(void);
int FIPS_rand_set_dt(unsigned char *dt);

int FIPS_rand_status(void);

const RAND_METHOD *FIPS_rand_method(void);

#ifdef  __cplusplus
}
#endif
#endif
#endif
