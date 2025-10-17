/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 8, 2023.
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

// cc_fuzz_target test for public key parsing.

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

extern "C" {

#include "includes.h"
#include "sshkey.h"
#include "ssherr.h"

static struct sshkey *generate_or_die(int type, unsigned bits) {
  int r;
  struct sshkey *ret;
  if ((r = sshkey_generate(type, bits, &ret)) != 0) {
    fprintf(stderr, "generate(%d, %u): %s", type, bits, ssh_err(r));
    abort();
  }
  return ret;
}

int LLVMFuzzerTestOneInput(const uint8_t* sig, size_t slen)
{
#ifdef WITH_OPENSSL
  static struct sshkey *rsa = generate_or_die(KEY_RSA, 2048);
  static struct sshkey *ecdsa256 = generate_or_die(KEY_ECDSA, 256);
  static struct sshkey *ecdsa384 = generate_or_die(KEY_ECDSA, 384);
  static struct sshkey *ecdsa521 = generate_or_die(KEY_ECDSA, 521);
#endif
  struct sshkey_sig_details *details = NULL;
  static struct sshkey *ed25519 = generate_or_die(KEY_ED25519, 0);
  static const char *data = "If everyone started announcing his nose had "
      "run away, I donâ€™t know how it would all end";
  static const size_t dlen = strlen(data);

#ifdef WITH_OPENSSL
  sshkey_verify(rsa, sig, slen, (const u_char *)data, dlen, NULL, 0, &details);
  sshkey_sig_details_free(details);
  details = NULL;

  sshkey_verify(ecdsa256, sig, slen, (const u_char *)data, dlen, NULL, 0, &details);
  sshkey_sig_details_free(details);
  details = NULL;

  sshkey_verify(ecdsa384, sig, slen, (const u_char *)data, dlen, NULL, 0, &details);
  sshkey_sig_details_free(details);
  details = NULL;

  sshkey_verify(ecdsa521, sig, slen, (const u_char *)data, dlen, NULL, 0, &details);
  sshkey_sig_details_free(details);
  details = NULL;
#endif

  sshkey_verify(ed25519, sig, slen, (const u_char *)data, dlen, NULL, 0, &details);
  sshkey_sig_details_free(details);
  return 0;
}

} // extern
