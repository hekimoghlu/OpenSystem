/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 4, 2022.
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

// cc_fuzz_target test for sshsig verification.

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

extern "C" {

#include "includes.h"
#include "sshkey.h"
#include "ssherr.h"
#include "sshbuf.h"
#include "sshsig.h"
#include "log.h"

int LLVMFuzzerTestOneInput(const uint8_t* sig, size_t slen)
{
  static const char *data = "If everyone started announcing his nose had "
      "run away, I donâ€™t know how it would all end";
  struct sshbuf *signature = sshbuf_from(sig, slen);
  struct sshbuf *message = sshbuf_from(data, strlen(data));
  struct sshkey *k = NULL;
  struct sshkey_sig_details *details = NULL;
  extern char *__progname;

  log_init(__progname, SYSLOG_LEVEL_QUIET, SYSLOG_FACILITY_USER, 1);
  sshsig_verifyb(signature, message, "castle", &k, &details);
  sshkey_sig_details_free(details);
  sshkey_free(k);
  sshbuf_free(signature);
  sshbuf_free(message);
  return 0;
}

} // extern
