/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 28, 2024.
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

#ifndef __crypto_header__
#define __crypto_header__

#ifndef PACKAGE_NAME
#error "need config.h"
#endif

#ifdef KRB5
#include <krb5-types.h>
#endif

#include <CommonCrypto/CommonDigest.h>
#include <CommonCrypto/CommonCryptor.h>
#include <CommonCrypto/CommonHMAC.h>

#include <CommonCrypto/CommonDigestSPI.h>
#include <CommonCrypto/CommonCryptorSPI.h>
#include <CommonCrypto/CommonRandomSPI.h>
#ifndef __APPLE_TARGET_EMBEDDED__
#include <hcrypto/des.h>
#include <hcrypto/rc4.h>
#include <hcrypto/rc2.h>
#include <hcrypto/rand.h>
#include <hcrypto/pkcs12.h>
#include <hcrypto/engine.h>
#include <hcrypto/hmac.h>
#endif

#include <hcrypto/evp.h>
#include <hcrypto/ui.h>
#include <hcrypto/ec.h>
#include <hcrypto/ecdsa.h>
#include <hcrypto/ecdh.h>

#ifndef CC_DIGEST_MAX_OUTPUT_SIZE
#define CC_DIGEST_MAX_OUTPUT_SIZE 128
#endif


#endif /* __crypto_header__ */
