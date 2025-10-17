/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 8, 2022.
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

int
otp_verify_user_1 (OtpContext *ctx, const char *passwd)
{
  OtpKey key1, key2;

  if (otp_parse (key1, passwd, ctx->alg)) {
    ctx->err = "Syntax error in reply";
    return -1;
  }
  memcpy (key2, key1, sizeof(key1));
  ctx->alg->next (key2);
  if (memcmp (ctx->key, key2, sizeof(key2)) == 0) {
    --ctx->n;
    memcpy (ctx->key, key1, sizeof(key1));
    return 0;
  } else
    return -1;
}

int
otp_verify_user (OtpContext *ctx, const char *passwd)
{
  void *dbm;
  int ret;

  if (!ctx->challengep)
    return -1;
  ret = otp_verify_user_1 (ctx, passwd);
  dbm = otp_db_open ();
  if (dbm == NULL) {
    free(ctx->user);
    return -1;
  }
  otp_put (dbm, ctx);
  free(ctx->user);
  otp_db_close (dbm);
  return ret;
}
