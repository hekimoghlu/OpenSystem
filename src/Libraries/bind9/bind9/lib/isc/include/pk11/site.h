/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 28, 2024.
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
/* The documentation about this file is in README.site */

#ifndef PK11_SITE_H
#define PK11_SITE_H 1

/*! \file pk11/site.h */

/*\brief Put here specific PKCS#11 tweaks
 *
 *\li PK11_<mechanism>_SKIP:
 *	Don't consider the lack of this mechanism as a fatal error.
 *
 *\li PK11_<mechanism>_REPLACE:
 *      Same as SKIP, and implement the mechanism using lower-level steps.
 *
 *\li PK11_<algorithm>_DISABLE:
 *	Same as SKIP, and disable support for the algorithm.
 */

/* current implemented flags are:
PK11_DH_PKCS_PARAMETER_GEN_SKIP
PK11_DSA_PARAMETER_GEN_SKIP
PK11_RSA_PKCS_REPLACE
PK11_MD5_HMAC_REPLACE
PK11_SHA_1_HMAC_REPLACE
PK11_SHA224_HMAC_REPLACE
PK11_SHA256_HMAC_REPLACE
PK11_SHA384_HMAC_REPLACE
PK11_SHA512_HMAC_REPLACE
PK11_MD5_DISABLE
PK11_DSA_DISABLE
PK11_DH_DISABLE
*/

/*
 * Predefined flavors
 */
/* Thales nCipher */
#define PK11_THALES_FLAVOR 0
/* SoftHSMv1 with SHA224 */
#define PK11_SOFTHSMV1_FLAVOR 1
/* SoftHSMv2 */
#define PK11_SOFTHSMV2_FLAVOR 2
/* Cryptech */
#define PK11_CRYPTECH_FLAVOR 3
/* AEP Keyper */
#define PK11_AEP_FLAVOR 4

/* Default is for Thales nCipher */
#ifndef PK11_FLAVOR
#define PK11_FLAVOR PK11_THALES_FLAVOR
#endif

#if PK11_FLAVOR == PK11_THALES_FLAVOR
#define PK11_DH_PKCS_PARAMETER_GEN_SKIP
/* doesn't work but supported #define PK11_DSA_PARAMETER_GEN_SKIP */
#define PK11_MD5_HMAC_REPLACE
#endif

#if PK11_FLAVOR == PK11_SOFTHSMV1_FLAVOR
#define PK11_DH_DISABLE
#define PK11_DSA_DISABLE
#define PK11_MD5_HMAC_REPLACE
#define PK11_SHA_1_HMAC_REPLACE
#define PK11_SHA224_HMAC_REPLACE
#define PK11_SHA256_HMAC_REPLACE
#define PK11_SHA384_HMAC_REPLACE
#define PK11_SHA512_HMAC_REPLACE
#endif

#if PK11_FLAVOR == PK11_SOFTHSMV2_FLAVOR
#endif

#if PK11_FLAVOR == PK11_CRYPTECH_FLAVOR
#define PK11_DH_DISABLE
#define PK11_DSA_DISABLE
#define PK11_MD5_DISABLE
#define PK11_SHA_1_HMAC_REPLACE
#define PK11_SHA224_HMAC_REPLACE
#define PK11_SHA256_HMAC_REPLACE
#define PK11_SHA384_HMAC_REPLACE
#define PK11_SHA512_HMAC_REPLACE
#endif

#if PK11_FLAVOR == PK11_AEP_FLAVOR
#define PK11_DH_DISABLE
#define PK11_DSA_DISABLE
#define PK11_RSA_PKCS_REPLACE
#define PK11_MD5_HMAC_REPLACE
#define PK11_SHA_1_HMAC_REPLACE
#define PK11_SHA224_HMAC_REPLACE
#define PK11_SHA256_HMAC_REPLACE
#define PK11_SHA384_HMAC_REPLACE
#define PK11_SHA512_HMAC_REPLACE
#endif

#endif /* PK11_SITE_H */
