/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 10, 2021.
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
#include "curl_setup.h"

#include <curl/curl.h>

#ifdef USE_LIBPSL

#include "psl.h"
#include "share.h"

/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

void Curl_psl_destroy(struct PslCache *pslcache)
{
  if(pslcache->psl) {
    if(pslcache->dynamic)
      psl_free((psl_ctx_t *) pslcache->psl);
    pslcache->psl = NULL;
    pslcache->dynamic = FALSE;
  }
}

static time_t now_seconds(void)
{
  struct curltime now = Curl_now();

  return now.tv_sec;
}

const psl_ctx_t *Curl_psl_use(struct Curl_easy *easy)
{
  struct PslCache *pslcache = easy->psl;
  const psl_ctx_t *psl;
  time_t now;

  if(!pslcache)
    return NULL;

  Curl_share_lock(easy, CURL_LOCK_DATA_PSL, CURL_LOCK_ACCESS_SHARED);
  now = now_seconds();
  if(!pslcache->psl || pslcache->expires <= now) {
    /* Let a chance to other threads to do the job: avoids deadlock. */
    Curl_share_unlock(easy, CURL_LOCK_DATA_PSL);

    /* Update cache: this needs an exclusive lock. */
    Curl_share_lock(easy, CURL_LOCK_DATA_PSL, CURL_LOCK_ACCESS_SINGLE);

    /* Recheck in case another thread did the job. */
    now = now_seconds();
    if(!pslcache->psl || pslcache->expires <= now) {
      bool dynamic = FALSE;
      time_t expires = TIME_T_MAX;

#if defined(PSL_VERSION_NUMBER) && PSL_VERSION_NUMBER >= 0x001000
      psl = psl_latest(NULL);
      dynamic = psl != NULL;
      /* Take care of possible time computation overflow. */
      expires = now < TIME_T_MAX - PSL_TTL? now + PSL_TTL: TIME_T_MAX;

      /* Only get the built-in PSL if we do not already have the "latest". */
      if(!psl && !pslcache->dynamic)
#endif

        psl = psl_builtin();

      if(psl) {
        Curl_psl_destroy(pslcache);
        pslcache->psl = psl;
        pslcache->dynamic = dynamic;
        pslcache->expires = expires;
      }
    }
    Curl_share_unlock(easy, CURL_LOCK_DATA_PSL);  /* Release exclusive lock. */
    Curl_share_lock(easy, CURL_LOCK_DATA_PSL, CURL_LOCK_ACCESS_SHARED);
  }
  psl = pslcache->psl;
  if(!psl)
    Curl_share_unlock(easy, CURL_LOCK_DATA_PSL);
  return psl;
}

void Curl_psl_release(struct Curl_easy *easy)
{
  Curl_share_unlock(easy, CURL_LOCK_DATA_PSL);
}

#endif /* USE_LIBPSL */
