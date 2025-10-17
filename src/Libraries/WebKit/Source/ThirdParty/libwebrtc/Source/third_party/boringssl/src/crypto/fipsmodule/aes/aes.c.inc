/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 2, 2023.
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
#include <openssl/aes.h>

#include <assert.h>

#include "internal.h"
#include "../modes/internal.h"


// Be aware that different sets of AES functions use incompatible key
// representations, varying in format of the key schedule, the |AES_KEY.rounds|
// value, or both. Therefore they cannot mix. Also, on AArch64, the plain-C
// code, above, is incompatible with the |aes_hw_*| functions.

void AES_encrypt(const uint8_t *in, uint8_t *out, const AES_KEY *key) {
  if (hwaes_capable()) {
    aes_hw_encrypt(in, out, key);
  } else if (vpaes_capable()) {
    vpaes_encrypt(in, out, key);
  } else {
    aes_nohw_encrypt(in, out, key);
  }
}

void AES_decrypt(const uint8_t *in, uint8_t *out, const AES_KEY *key) {
  if (hwaes_capable()) {
    aes_hw_decrypt(in, out, key);
  } else if (vpaes_capable()) {
    vpaes_decrypt(in, out, key);
  } else {
    aes_nohw_decrypt(in, out, key);
  }
}

int AES_set_encrypt_key(const uint8_t *key, unsigned bits, AES_KEY *aeskey) {
  if (bits != 128 && bits != 192 && bits != 256) {
    return -2;
  }
  if (hwaes_capable()) {
    return aes_hw_set_encrypt_key(key, bits, aeskey);
  } else if (vpaes_capable()) {
    return vpaes_set_encrypt_key(key, bits, aeskey);
  } else {
    return aes_nohw_set_encrypt_key(key, bits, aeskey);
  }
}

int AES_set_decrypt_key(const uint8_t *key, unsigned bits, AES_KEY *aeskey) {
  if (bits != 128 && bits != 192 && bits != 256) {
    return -2;
  }
  if (hwaes_capable()) {
    return aes_hw_set_decrypt_key(key, bits, aeskey);
  } else if (vpaes_capable()) {
    return vpaes_set_decrypt_key(key, bits, aeskey);
  } else {
    return aes_nohw_set_decrypt_key(key, bits, aeskey);
  }
}

#if defined(HWAES) && (defined(OPENSSL_X86) || defined(OPENSSL_X86_64))
// On x86 and x86_64, |aes_hw_set_decrypt_key|, we implement
// |aes_hw_encrypt_key_to_decrypt_key| in assembly and rely on C code to combine
// the operations.
int aes_hw_set_decrypt_key(const uint8_t *user_key, int bits, AES_KEY *key) {
  int ret = aes_hw_set_encrypt_key(user_key, bits, key);
  if (ret == 0) {
    aes_hw_encrypt_key_to_decrypt_key(key);
  }
  return ret;
}

int aes_hw_set_encrypt_key(const uint8_t *user_key, int bits, AES_KEY *key) {
  if (aes_hw_set_encrypt_key_alt_preferred()) {
    return aes_hw_set_encrypt_key_alt(user_key, bits, key);
  } else {
    return aes_hw_set_encrypt_key_base(user_key, bits, key);
  }
}
#endif
