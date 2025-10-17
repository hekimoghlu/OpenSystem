/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 17, 2025.
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
#include <CNIOBoringSSL_asn1.h>
#include <CNIOBoringSSL_dsa.h>
#include <CNIOBoringSSL_ec_key.h>
#include <CNIOBoringSSL_err.h>
#include <CNIOBoringSSL_evp.h>
#include <CNIOBoringSSL_rsa.h>


int i2d_PrivateKey(const EVP_PKEY *a, uint8_t **pp) {
  switch (EVP_PKEY_id(a)) {
    case EVP_PKEY_RSA:
      return i2d_RSAPrivateKey(EVP_PKEY_get0_RSA(a), pp);
    case EVP_PKEY_EC:
      return i2d_ECPrivateKey(EVP_PKEY_get0_EC_KEY(a), pp);
    case EVP_PKEY_DSA:
      return i2d_DSAPrivateKey(EVP_PKEY_get0_DSA(a), pp);
    default:
      // Although this file is in crypto/x509 for layering reasons, it emits
      // an error code from ASN1 for OpenSSL compatibility.
      OPENSSL_PUT_ERROR(ASN1, ASN1_R_UNSUPPORTED_PUBLIC_KEY_TYPE);
      return -1;
  }
}
