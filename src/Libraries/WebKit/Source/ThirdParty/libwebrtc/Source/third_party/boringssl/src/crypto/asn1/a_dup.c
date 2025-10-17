/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 4, 2025.
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
#include <openssl/asn1.h>

#include <openssl/err.h>
#include <openssl/mem.h>

// ASN1_ITEM version of dup: this follows the model above except we don't
// need to allocate the buffer. At some point this could be rewritten to
// directly dup the underlying structure instead of doing and encode and
// decode.
void *ASN1_item_dup(const ASN1_ITEM *it, void *x) {
  unsigned char *b = NULL;
  const unsigned char *p;
  long i;
  void *ret;

  if (x == NULL) {
    return NULL;
  }

  i = ASN1_item_i2d(x, &b, it);
  if (b == NULL) {
    return NULL;
  }
  p = b;
  ret = ASN1_item_d2i(NULL, &p, i, it);
  OPENSSL_free(b);
  return ret;
}
