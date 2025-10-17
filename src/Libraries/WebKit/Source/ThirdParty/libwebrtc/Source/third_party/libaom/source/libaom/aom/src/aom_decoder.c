/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 22, 2023.
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
/*!\file
 * \brief Provides the high level interface to wrap decoder algorithms.
 *
 */
#include <string.h>
#include "aom/internal/aom_codec_internal.h"

#define SAVE_STATUS(ctx, var) (ctx ? (ctx->err = var) : var)

static aom_codec_alg_priv_t *get_alg_priv(aom_codec_ctx_t *ctx) {
  return (aom_codec_alg_priv_t *)ctx->priv;
}

aom_codec_err_t aom_codec_dec_init_ver(aom_codec_ctx_t *ctx,
                                       aom_codec_iface_t *iface,
                                       const aom_codec_dec_cfg_t *cfg,
                                       aom_codec_flags_t flags, int ver) {
  aom_codec_err_t res;

  if (ver != AOM_DECODER_ABI_VERSION)
    res = AOM_CODEC_ABI_MISMATCH;
  else if (!ctx || !iface)
    res = AOM_CODEC_INVALID_PARAM;
  else if (iface->abi_version != AOM_CODEC_INTERNAL_ABI_VERSION)
    res = AOM_CODEC_ABI_MISMATCH;
  else if (!(iface->caps & AOM_CODEC_CAP_DECODER))
    res = AOM_CODEC_INCAPABLE;
  else {
    memset(ctx, 0, sizeof(*ctx));
    ctx->iface = iface;
    ctx->name = iface->name;
    ctx->priv = NULL;
    ctx->init_flags = flags;
    ctx->config.dec = cfg;

    res = ctx->iface->init(ctx);
    if (res) {
      ctx->err_detail = ctx->priv ? ctx->priv->err_detail : NULL;
      aom_codec_destroy(ctx);
    }
  }

  return SAVE_STATUS(ctx, res);
}

aom_codec_err_t aom_codec_peek_stream_info(aom_codec_iface_t *iface,
                                           const uint8_t *data, size_t data_sz,
                                           aom_codec_stream_info_t *si) {
  aom_codec_err_t res;

  if (!iface || !data || !data_sz || !si) {
    res = AOM_CODEC_INVALID_PARAM;
  } else {
    /* Set default/unknown values */
    si->w = 0;
    si->h = 0;

    res = iface->dec.peek_si(data, data_sz, si);
  }

  return res;
}

aom_codec_err_t aom_codec_get_stream_info(aom_codec_ctx_t *ctx,
                                          aom_codec_stream_info_t *si) {
  aom_codec_err_t res;

  if (!ctx || !si) {
    res = AOM_CODEC_INVALID_PARAM;
  } else if (!ctx->iface || !ctx->priv) {
    res = AOM_CODEC_ERROR;
  } else {
    /* Set default/unknown values */
    si->w = 0;
    si->h = 0;

    res = ctx->iface->dec.get_si(get_alg_priv(ctx), si);
  }

  return SAVE_STATUS(ctx, res);
}

aom_codec_err_t aom_codec_decode(aom_codec_ctx_t *ctx, const uint8_t *data,
                                 size_t data_sz, void *user_priv) {
  aom_codec_err_t res;

  if (!ctx)
    res = AOM_CODEC_INVALID_PARAM;
  else if (!ctx->iface || !ctx->priv)
    res = AOM_CODEC_ERROR;
  else {
    res = ctx->iface->dec.decode(get_alg_priv(ctx), data, data_sz, user_priv);
  }

  return SAVE_STATUS(ctx, res);
}

aom_image_t *aom_codec_get_frame(aom_codec_ctx_t *ctx, aom_codec_iter_t *iter) {
  aom_image_t *img;

  if (!ctx || !iter || !ctx->iface || !ctx->priv)
    img = NULL;
  else
    img = ctx->iface->dec.get_frame(get_alg_priv(ctx), iter);

  return img;
}

aom_codec_err_t aom_codec_set_frame_buffer_functions(
    aom_codec_ctx_t *ctx, aom_get_frame_buffer_cb_fn_t cb_get,
    aom_release_frame_buffer_cb_fn_t cb_release, void *cb_priv) {
  aom_codec_err_t res;

  if (!ctx || !cb_get || !cb_release) {
    res = AOM_CODEC_INVALID_PARAM;
  } else if (!ctx->iface || !ctx->priv) {
    res = AOM_CODEC_ERROR;
  } else if (!(ctx->iface->caps & AOM_CODEC_CAP_EXTERNAL_FRAME_BUFFER)) {
    res = AOM_CODEC_INCAPABLE;
  } else {
    res = ctx->iface->dec.set_fb_fn(get_alg_priv(ctx), cb_get, cb_release,
                                    cb_priv);
  }

  return SAVE_STATUS(ctx, res);
}
