/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 21, 2022.
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
/*
 * $Id$
 */

#ifndef _HEIM_PKCS12_H
#define _HEIM_PKCS12_H 1

/* symbol renaming */
#define PKCS12_key_gen hc_PKCS12_key_gen

/*
 *
 */

#include <hcrypto/evp.h>

#define PKCS12_KEY_ID 1
#define PKCS12_IV_ID 2

int	PKCS12_key_gen(const void *, size_t, const void *,
		       size_t, int, int, size_t, void *, const EVP_MD *);


#endif /* _HEIM_PKCS12_H */
