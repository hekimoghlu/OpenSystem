/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 23, 2023.
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

#if !defined(CURL_DISABLE_PROXY)

#include <curl/curl.h>
#include "urldata.h"
#include "cfilters.h"
#include "cf-haproxy.h"
#include "curl_trc.h"
#include "multiif.h"

/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"


typedef enum {
    HAPROXY_INIT,     /* init/default/no tunnel state */
    HAPROXY_SEND,     /* data_out being sent */
    HAPROXY_DONE      /* all work done */
} haproxy_state;

struct cf_haproxy_ctx {
  int state;
  struct dynbuf data_out;
};

static void cf_haproxy_ctx_reset(struct cf_haproxy_ctx *ctx)
{
  DEBUGASSERT(ctx);
  ctx->state = HAPROXY_INIT;
  Curl_dyn_reset(&ctx->data_out);
}

static void cf_haproxy_ctx_free(struct cf_haproxy_ctx *ctx)
{
  if(ctx) {
    Curl_dyn_free(&ctx->data_out);
    free(ctx);
  }
}

static CURLcode cf_haproxy_date_out_set(struct Curl_cfilter*cf,
                                        struct Curl_easy *data)
{
  struct cf_haproxy_ctx *ctx = cf->ctx;
  CURLcode result;
  const char *tcp_version;
  const char *client_ip;

  DEBUGASSERT(ctx);
  DEBUGASSERT(ctx->state == HAPROXY_INIT);
#ifdef USE_UNIX_SOCKETS
  if(cf->conn->unix_domain_socket)
    /* the buffer is large enough to hold this! */
    result = Curl_dyn_addn(&ctx->data_out, STRCONST("PROXY UNKNOWN\r\n"));
  else {
#endif /* USE_UNIX_SOCKETS */
  /* Emit the correct prefix for IPv6 */
  tcp_version = cf->conn->bits.ipv6 ? "TCP6" : "TCP4";
  if(data->set.str[STRING_HAPROXY_CLIENT_IP])
    client_ip = data->set.str[STRING_HAPROXY_CLIENT_IP];
  else
    client_ip = data->info.primary.local_ip;

  result = Curl_dyn_addf(&ctx->data_out, "PROXY %s %s %s %i %i\r\n",
                         tcp_version,
                         client_ip,
                         data->info.primary.remote_ip,
                         data->info.primary.local_port,
                         data->info.primary.remote_port);

#ifdef USE_UNIX_SOCKETS
  }
#endif /* USE_UNIX_SOCKETS */
  return result;
}

static CURLcode cf_haproxy_connect(struct Curl_cfilter *cf,
                                   struct Curl_easy *data,
                                   bool blocking, bool *done)
{
  struct cf_haproxy_ctx *ctx = cf->ctx;
  CURLcode result;
  size_t len;

  DEBUGASSERT(ctx);
  if(cf->connected) {
    *done = TRUE;
    return CURLE_OK;
  }

  result = cf->next->cft->do_connect(cf->next, data, blocking, done);
  if(result || !*done)
    return result;

  switch(ctx->state) {
  case HAPROXY_INIT:
    result = cf_haproxy_date_out_set(cf, data);
    if(result)
      goto out;
    ctx->state = HAPROXY_SEND;
    FALLTHROUGH();
  case HAPROXY_SEND:
    len = Curl_dyn_len(&ctx->data_out);
    if(len > 0) {
      size_t written;
      result = Curl_conn_send(data, cf->sockindex,
                              Curl_dyn_ptr(&ctx->data_out),
                              len, &written);
      if(result == CURLE_AGAIN) {
        result = CURLE_OK;
        written = 0;
      }
      else if(result)
        goto out;
      Curl_dyn_tail(&ctx->data_out, len - written);
      if(Curl_dyn_len(&ctx->data_out) > 0) {
        result = CURLE_OK;
        goto out;
      }
    }
    ctx->state = HAPROXY_DONE;
    FALLTHROUGH();
  default:
    Curl_dyn_free(&ctx->data_out);
    break;
  }

out:
  *done = (!result) && (ctx->state == HAPROXY_DONE);
  cf->connected = *done;
  return result;
}

static void cf_haproxy_destroy(struct Curl_cfilter *cf,
                               struct Curl_easy *data)
{
  (void)data;
  CURL_TRC_CF(data, cf, "destroy");
  cf_haproxy_ctx_free(cf->ctx);
}

static void cf_haproxy_close(struct Curl_cfilter *cf,
                             struct Curl_easy *data)
{
  CURL_TRC_CF(data, cf, "close");
  cf->connected = FALSE;
  cf_haproxy_ctx_reset(cf->ctx);
  if(cf->next)
    cf->next->cft->do_close(cf->next, data);
}

static void cf_haproxy_adjust_pollset(struct Curl_cfilter *cf,
                                      struct Curl_easy *data,
                                      struct easy_pollset *ps)
{
  if(cf->next->connected && !cf->connected) {
    /* If we are not connected, but the filter "below" is
     * and not waiting on something, we are sending. */
    Curl_pollset_set_out_only(data, ps, Curl_conn_cf_get_socket(cf, data));
  }
}

struct Curl_cftype Curl_cft_haproxy = {
  "HAPROXY",
  0,
  0,
  cf_haproxy_destroy,
  cf_haproxy_connect,
  cf_haproxy_close,
  Curl_cf_def_get_host,
  cf_haproxy_adjust_pollset,
  Curl_cf_def_data_pending,
  Curl_cf_def_send,
  Curl_cf_def_recv,
  Curl_cf_def_cntrl,
  Curl_cf_def_conn_is_alive,
  Curl_cf_def_conn_keep_alive,
  Curl_cf_def_query,
};

static CURLcode cf_haproxy_create(struct Curl_cfilter **pcf,
                                  struct Curl_easy *data)
{
  struct Curl_cfilter *cf = NULL;
  struct cf_haproxy_ctx *ctx;
  CURLcode result;

  (void)data;
  ctx = calloc(1, sizeof(*ctx));
  if(!ctx) {
    result = CURLE_OUT_OF_MEMORY;
    goto out;
  }
  ctx->state = HAPROXY_INIT;
  Curl_dyn_init(&ctx->data_out, DYN_HAXPROXY);

  result = Curl_cf_create(&cf, &Curl_cft_haproxy, ctx);
  if(result)
    goto out;
  ctx = NULL;

out:
  cf_haproxy_ctx_free(ctx);
  *pcf = result? NULL : cf;
  return result;
}

CURLcode Curl_cf_haproxy_insert_after(struct Curl_cfilter *cf_at,
                                      struct Curl_easy *data)
{
  struct Curl_cfilter *cf;
  CURLcode result;

  result = cf_haproxy_create(&cf, data);
  if(result)
    goto out;
  Curl_conn_cf_insert_after(cf_at, cf);

out:
  return result;
}

#endif /* !CURL_DISABLE_PROXY */

