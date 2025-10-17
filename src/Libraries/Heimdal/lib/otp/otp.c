/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 6, 2025.
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
#ifdef HAVE_CONFIG_H
#include "config.h"
RCSID("$Id$");
#endif

#include "otp_locl.h"
#include "otp_md.h"

static OtpAlgorithm algorithms[] = {
  {OTP_ALG_MD4, "md4", 16, otp_md4_hash, otp_md4_init, otp_md4_next},
  {OTP_ALG_MD5, "md5", 16, otp_md5_hash, otp_md5_init, otp_md5_next},
  {OTP_ALG_SHA, "sha", 20, otp_sha_hash, otp_sha_init, otp_sha_next}
};

OtpAlgorithm *
otp_find_alg (char *name)
{
  int i;

  for (i = 0; i < sizeof(algorithms)/sizeof(*algorithms); ++i)
    if (strcmp (name, algorithms[i].name) == 0)
      return &algorithms[i];
  return NULL;
}

char *
otp_error (OtpContext *o)
{
  return o->err;
}
