/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 24, 2024.
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
#include <openssl/bn.h>
#include <openssl/dsa.h>


struct wrapped_callback {
  void (*callback)(int, int, void *);
  void *arg;
};

// callback_wrapper converts an â€œoldâ€ style generation callback to the newer
// |BN_GENCB| form.
static int callback_wrapper(int event, int n, BN_GENCB *gencb) {
  struct wrapped_callback *wrapped = (struct wrapped_callback *) gencb->arg;
  wrapped->callback(event, n, wrapped->arg);
  return 1;
}

DSA *DSA_generate_parameters(int bits, uint8_t *seed_in, int seed_len,
                             int *counter_ret, unsigned long *h_ret,
                             void (*callback)(int, int, void *), void *cb_arg) {
  if (bits < 0 || seed_len < 0) {
      return NULL;
  }

  DSA *ret = DSA_new();
  if (ret == NULL) {
      return NULL;
  }

  BN_GENCB gencb_storage;
  BN_GENCB *cb = NULL;

  struct wrapped_callback wrapped;

  if (callback != NULL) {
    wrapped.callback = callback;
    wrapped.arg = cb_arg;

    cb = &gencb_storage;
    BN_GENCB_set(cb, callback_wrapper, &wrapped);
  }

  if (!DSA_generate_parameters_ex(ret, bits, seed_in, seed_len, counter_ret,
                                  h_ret, cb)) {
    goto err;
  }

  return ret;

err:
  DSA_free(ret);
  return NULL;
}
