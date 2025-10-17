/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 29, 2021.
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
#include <assert.h>
#include <stdlib.h>
#include "vpx_config.h"
#include "lookahead.h"
#include "vp8/common/extend.h"

#define MAX_LAG_BUFFERS (CONFIG_REALTIME_ONLY ? 1 : 25)

struct lookahead_ctx {
  unsigned int max_sz;         /* Absolute size of the queue */
  unsigned int sz;             /* Number of buffers currently in the queue */
  unsigned int read_idx;       /* Read index */
  unsigned int write_idx;      /* Write index */
  struct lookahead_entry *buf; /* Buffer list */
};

/* Return the buffer at the given absolute index and increment the index */
static struct lookahead_entry *pop(struct lookahead_ctx *ctx,
                                   unsigned int *idx) {
  unsigned int index = *idx;
  struct lookahead_entry *buf = ctx->buf + index;

  assert(index < ctx->max_sz);
  if (++index >= ctx->max_sz) index -= ctx->max_sz;
  *idx = index;
  return buf;
}

void vp8_lookahead_destroy(struct lookahead_ctx *ctx) {
  if (ctx) {
    if (ctx->buf) {
      unsigned int i;

      for (i = 0; i < ctx->max_sz; ++i) {
        vp8_yv12_de_alloc_frame_buffer(&ctx->buf[i].img);
      }
      free(ctx->buf);
    }
    free(ctx);
  }
}

struct lookahead_ctx *vp8_lookahead_init(unsigned int width,
                                         unsigned int height,
                                         unsigned int depth) {
  struct lookahead_ctx *ctx = NULL;
  unsigned int i;

  /* Clamp the lookahead queue depth */
  if (depth < 1) {
    depth = 1;
  } else if (depth > MAX_LAG_BUFFERS) {
    depth = MAX_LAG_BUFFERS;
  }

  /* Keep last frame in lookahead buffer by increasing depth by 1.*/
  depth += 1;

  /* Align the buffer dimensions */
  width = (width + 15) & ~15u;
  height = (height + 15) & ~15u;

  /* Allocate the lookahead structures */
  ctx = calloc(1, sizeof(*ctx));
  if (ctx) {
    ctx->max_sz = depth;
    ctx->buf = calloc(depth, sizeof(*ctx->buf));
    if (!ctx->buf) goto bail;
    for (i = 0; i < depth; ++i) {
      if (vp8_yv12_alloc_frame_buffer(&ctx->buf[i].img, width, height,
                                      VP8BORDERINPIXELS)) {
        goto bail;
      }
    }
  }
  return ctx;
bail:
  vp8_lookahead_destroy(ctx);
  return NULL;
}

int vp8_lookahead_push(struct lookahead_ctx *ctx, YV12_BUFFER_CONFIG *src,
                       int64_t ts_start, int64_t ts_end, unsigned int flags,
                       unsigned char *active_map) {
  struct lookahead_entry *buf;
  int row, col, active_end;
  int mb_rows = (src->y_height + 15) >> 4;
  int mb_cols = (src->y_width + 15) >> 4;

  if (ctx->sz + 2 > ctx->max_sz) return 1;
  ctx->sz++;
  buf = pop(ctx, &ctx->write_idx);

  /* Only do this partial copy if the following conditions are all met:
   * 1. Lookahead queue has has size of 1.
   * 2. Active map is provided.
   * 3. This is not a key frame, golden nor altref frame.
   */
  if (ctx->max_sz == 1 && active_map && !flags) {
    for (row = 0; row < mb_rows; ++row) {
      col = 0;

      while (1) {
        /* Find the first active macroblock in this row. */
        for (; col < mb_cols; ++col) {
          if (active_map[col]) break;
        }

        /* No more active macroblock in this row. */
        if (col == mb_cols) break;

        /* Find the end of active region in this row. */
        active_end = col;

        for (; active_end < mb_cols; ++active_end) {
          if (!active_map[active_end]) break;
        }

        /* Only copy this active region. */
        vp8_copy_and_extend_frame_with_rect(src, &buf->img, row << 4, col << 4,
                                            16, (active_end - col) << 4);

        /* Start again from the end of this active region. */
        col = active_end;
      }

      active_map += mb_cols;
    }
  } else {
    vp8_copy_and_extend_frame(src, &buf->img);
  }
  buf->ts_start = ts_start;
  buf->ts_end = ts_end;
  buf->flags = flags;
  return 0;
}

struct lookahead_entry *vp8_lookahead_pop(struct lookahead_ctx *ctx,
                                          int drain) {
  struct lookahead_entry *buf = NULL;

  assert(ctx != NULL);
  if (ctx->sz && (drain || ctx->sz == ctx->max_sz - 1)) {
    buf = pop(ctx, &ctx->read_idx);
    ctx->sz--;
  }
  return buf;
}

struct lookahead_entry *vp8_lookahead_peek(struct lookahead_ctx *ctx,
                                           unsigned int index, int direction) {
  struct lookahead_entry *buf = NULL;

  if (direction == PEEK_FORWARD) {
    assert(index < ctx->max_sz - 1);
    if (index < ctx->sz) {
      index += ctx->read_idx;
      if (index >= ctx->max_sz) index -= ctx->max_sz;
      buf = ctx->buf + index;
    }
  } else if (direction == PEEK_BACKWARD) {
    assert(index == 1);

    if (ctx->read_idx == 0) {
      index = ctx->max_sz - 1;
    } else {
      index = ctx->read_idx - index;
    }
    buf = ctx->buf + index;
  }

  return buf;
}

unsigned int vp8_lookahead_depth(struct lookahead_ctx *ctx) { return ctx->sz; }
