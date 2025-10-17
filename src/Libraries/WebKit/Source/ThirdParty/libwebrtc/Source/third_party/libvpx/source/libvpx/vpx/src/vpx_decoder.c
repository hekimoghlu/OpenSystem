/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 20, 2025.
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
#include "vpx/internal/vpx_codec_internal.h"

#define SAVE_STATUS(ctx, var) (ctx ? (ctx->err = var) : var)

static vpx_codec_alg_priv_t *get_alg_priv(vpx_codec_ctx_t *ctx) {
  return (vpx_codec_alg_priv_t *)ctx->priv;
}

vpx_codec_err_t vpx_codec_dec_init_ver(vpx_codec_ctx_t *ctx,
                                       vpx_codec_iface_t *iface,
                                       const vpx_codec_dec_cfg_t *cfg,
                                       vpx_codec_flags_t flags, int ver) {
  vpx_codec_err_t res;

  if (ver != VPX_DECODER_ABI_VERSION)
    res = VPX_CODEC_ABI_MISMATCH;
  else if (!ctx || !iface)
    res = VPX_CODEC_INVALID_PARAM;
  else if (iface->abi_version != VPX_CODEC_INTERNAL_ABI_VERSION)
    res = VPX_CODEC_ABI_MISMATCH;
  else if ((flags & VPX_CODEC_USE_POSTPROC) &&
           !(iface->caps & VPX_CODEC_CAP_POSTPROC))
    res = VPX_CODEC_INCAPABLE;
  else if ((flags & VPX_CODEC_USE_ERROR_CONCEALMENT) &&
           !(iface->caps & VPX_CODEC_CAP_ERROR_CONCEALMENT))
    res = VPX_CODEC_INCAPABLE;
  else if ((flags & VPX_CODEC_USE_INPUT_FRAGMENTS) &&
           !(iface->caps & VPX_CODEC_CAP_INPUT_FRAGMENTS))
    res = VPX_CODEC_INCAPABLE;
  else if (!(iface->caps & VPX_CODEC_CAP_DECODER))
    res = VPX_CODEC_INCAPABLE;
  else {
    memset(ctx, 0, sizeof(*ctx));
    ctx->iface = iface;
    ctx->name = iface->name;
    ctx->priv = NULL;
    ctx->init_flags = flags;
    ctx->config.dec = cfg;

    res = ctx->iface->init(ctx, NULL);
    if (res) {
      ctx->err_detail = ctx->priv ? ctx->priv->err_detail : NULL;
      vpx_codec_destroy(ctx);
    }
  }

  return SAVE_STATUS(ctx, res);
}

vpx_codec_err_t vpx_codec_peek_stream_info(vpx_codec_iface_t *iface,
                                           const uint8_t *data,
                                           unsigned int data_sz,
                                           vpx_codec_stream_info_t *si) {
  vpx_codec_err_t res;

  if (!iface || !data || !data_sz || !si ||
      si->sz < sizeof(vpx_codec_stream_info_t))
    res = VPX_CODEC_INVALID_PARAM;
  else {
    /* Set default/unknown values */
    si->w = 0;
    si->h = 0;

    res = iface->dec.peek_si(data, data_sz, si);
  }

  return res;
}

vpx_codec_err_t vpx_codec_get_stream_info(vpx_codec_ctx_t *ctx,
                                          vpx_codec_stream_info_t *si) {
  vpx_codec_err_t res;

  if (!ctx || !si || si->sz < sizeof(vpx_codec_stream_info_t))
    res = VPX_CODEC_INVALID_PARAM;
  else if (!ctx->iface || !ctx->priv)
    res = VPX_CODEC_ERROR;
  else {
    /* Set default/unknown values */
    si->w = 0;
    si->h = 0;

    res = ctx->iface->dec.get_si(get_alg_priv(ctx), si);
  }

  return SAVE_STATUS(ctx, res);
}

vpx_codec_err_t vpx_codec_decode(vpx_codec_ctx_t *ctx, const uint8_t *data,
                                 unsigned int data_sz, void *user_priv,
                                 long deadline) {
  vpx_codec_err_t res;
  (void)deadline;

  /* Sanity checks */
  /* NULL data ptr allowed if data_sz is 0 too */
  if (!ctx || (!data && data_sz) || (data && !data_sz))
    res = VPX_CODEC_INVALID_PARAM;
  else if (!ctx->iface || !ctx->priv)
    res = VPX_CODEC_ERROR;
  else
    res = ctx->iface->dec.decode(get_alg_priv(ctx), data, data_sz, user_priv);

  return SAVE_STATUS(ctx, res);
}

vpx_image_t *vpx_codec_get_frame(vpx_codec_ctx_t *ctx, vpx_codec_iter_t *iter) {
  vpx_image_t *img;

  if (!ctx || !iter || !ctx->iface || !ctx->priv)
    img = NULL;
  else
    img = ctx->iface->dec.get_frame(get_alg_priv(ctx), iter);

  return img;
}

vpx_codec_err_t vpx_codec_register_put_frame_cb(vpx_codec_ctx_t *ctx,
                                                vpx_codec_put_frame_cb_fn_t cb,
                                                void *user_priv) {
  vpx_codec_err_t res;

  if (!ctx || !cb)
    res = VPX_CODEC_INVALID_PARAM;
  else if (!ctx->iface || !ctx->priv)
    res = VPX_CODEC_ERROR;
  else if (!(ctx->iface->caps & VPX_CODEC_CAP_PUT_FRAME))
    res = VPX_CODEC_INCAPABLE;
  else {
    ctx->priv->dec.put_frame_cb.u.put_frame = cb;
    ctx->priv->dec.put_frame_cb.user_priv = user_priv;
    res = VPX_CODEC_OK;
  }

  return SAVE_STATUS(ctx, res);
}

vpx_codec_err_t vpx_codec_register_put_slice_cb(vpx_codec_ctx_t *ctx,
                                                vpx_codec_put_slice_cb_fn_t cb,
                                                void *user_priv) {
  vpx_codec_err_t res;

  if (!ctx || !cb)
    res = VPX_CODEC_INVALID_PARAM;
  else if (!ctx->iface || !ctx->priv)
    res = VPX_CODEC_ERROR;
  else if (!(ctx->iface->caps & VPX_CODEC_CAP_PUT_SLICE))
    res = VPX_CODEC_INCAPABLE;
  else {
    ctx->priv->dec.put_slice_cb.u.put_slice = cb;
    ctx->priv->dec.put_slice_cb.user_priv = user_priv;
    res = VPX_CODEC_OK;
  }

  return SAVE_STATUS(ctx, res);
}

vpx_codec_err_t vpx_codec_set_frame_buffer_functions(
    vpx_codec_ctx_t *ctx, vpx_get_frame_buffer_cb_fn_t cb_get,
    vpx_release_frame_buffer_cb_fn_t cb_release, void *cb_priv) {
  vpx_codec_err_t res;

  if (!ctx || !cb_get || !cb_release) {
    res = VPX_CODEC_INVALID_PARAM;
  } else if (!ctx->iface || !ctx->priv) {
    res = VPX_CODEC_ERROR;
  } else if (!(ctx->iface->caps & VPX_CODEC_CAP_EXTERNAL_FRAME_BUFFER)) {
    res = VPX_CODEC_INCAPABLE;
  } else {
    res = ctx->iface->dec.set_fb_fn(get_alg_priv(ctx), cb_get, cb_release,
                                    cb_priv);
  }

  return SAVE_STATUS(ctx, res);
}
