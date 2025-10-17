/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 14, 2022.
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

#include <stdlib.h>
#include <stdio.h>

#include "jv_thread.h"
#include "jv_dtoa_tsd.h"
#include "jv_dtoa.h"
#include "jv_alloc.h"

static pthread_once_t dtoa_ctx_once = PTHREAD_ONCE_INIT;

static pthread_key_t dtoa_ctx_key;
static void tsd_dtoa_ctx_dtor(void *ctx) {
  if (ctx) {
    jvp_dtoa_context_free((struct dtoa_context *)ctx);
    jv_mem_free(ctx);
  }
}

#ifndef WIN32
static
#endif
void jv_tsd_dtoa_ctx_fini() {
  struct dtoa_context *ctx = pthread_getspecific(dtoa_ctx_key);
  tsd_dtoa_ctx_dtor(ctx);
  pthread_setspecific(dtoa_ctx_key, NULL);
}

#ifndef WIN32
static
#endif
void jv_tsd_dtoa_ctx_init() {
  if (pthread_key_create(&dtoa_ctx_key, tsd_dtoa_ctx_dtor) != 0) {
    fprintf(stderr, "error: cannot create thread specific key");
    abort();
  }
  atexit(jv_tsd_dtoa_ctx_fini);
}

inline struct dtoa_context *tsd_dtoa_context_get() {
  pthread_once(&dtoa_ctx_once, jv_tsd_dtoa_ctx_init); // cannot fail
  struct dtoa_context *ctx = (struct dtoa_context*)pthread_getspecific(dtoa_ctx_key);
  if (!ctx) {
    ctx = malloc(sizeof(struct dtoa_context));
    jvp_dtoa_context_init(ctx);
    if (pthread_setspecific(dtoa_ctx_key, ctx) != 0) {
      fprintf(stderr, "error: cannot set thread specific data");
      abort();
    }
  }
  return ctx;
}
