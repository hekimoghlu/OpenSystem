/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 27, 2022.
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
otp_challenge (OtpContext *ctx, char *user, char *str, size_t len)
{
  void *dbm;
  int ret;

  ctx->challengep = 0;
  ctx->err = NULL;
  ctx->user = strdup(user);
  if (ctx->user == NULL) {
    ctx->err = "Out of memory";
    return -1;
  }
  dbm = otp_db_open ();
  if (dbm == NULL) {
    ctx->err = "Cannot open database";
    return -1;
  }
  ret = otp_get (dbm, ctx);
  otp_db_close (dbm);
  if (ret)
    return ret;
  snprintf (str, len,
	    "[ otp-%s %u %s ]",
	    ctx->alg->name, ctx->n-1, ctx->seed);
  ctx->challengep = 1;
  return 0;
}
