/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 21, 2023.
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
#include <config.h>

#define HC_DEPRECATED

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <krb5-types.h>
#include <assert.h>

#include <CommonCrypto/CommonCryptor.h>
#ifndef __APPLE_TARGET_EMBEDDED__
#include <CommonCrypto/CommonCryptorSPI.h>
#else
#include "CCDGlue.h"
#endif


#include <roken.h>

#include "des.h"


void
DES_set_odd_parity(DES_cblock *key)
{
    CCDesSetOddParity(key, sizeof(*key));
}

int
DES_is_weak_key(DES_cblock *key)
{
    if (CCDesIsWeakKey(key, sizeof(*key)))
	return 1;
    return 0;
}
