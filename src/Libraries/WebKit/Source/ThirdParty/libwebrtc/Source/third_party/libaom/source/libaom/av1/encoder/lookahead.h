/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 4, 2022.
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
 * \brief Describes look ahead buffer operations.
 */
#ifndef AOM_AV1_ENCODER_LOOKAHEAD_H_
#define AOM_AV1_ENCODER_LOOKAHEAD_H_

#include <stdbool.h>

#include "aom_scale/yv12config.h"
#include "aom/aom_integer.h"

#ifdef __cplusplus
extern "C" {
#endif

/*!\cond */
#define MAX_LAG_BUFFERS 48
#define MAX_LAP_BUFFERS 48
#define MAX_TOTAL_BUFFERS (MAX_LAG_BUFFERS + MAX_LAP_BUFFERS)
#define LAP_LAG_IN_FRAMES 17

struct lookahead_entry {
  YV12_BUFFER_CONFIG img;
  int64_t ts_start;
  int64_t ts_end;
  int display_idx;
  aom_enc_frame_flags_t flags;
};

// The max of past frames we want to keep in the queue.
#define MAX_PRE_FRAMES 1

enum { ENCODE_STAGE, LAP_STAGE, MAX_STAGES } UENUM1BYTE(COMPRESSOR_STAGE);

struct read_ctx {
  int sz;       /* Number of buffers currently in the queue */
  int read_idx; /* Read index */
  int pop_sz;   /* Size to check for pop condition */
  int valid;    /* Is this ctx valid? */
};

struct lookahead_ctx {
  int max_sz;                            /* Absolute size of the queue */
  int write_idx;                         /* Write index */
  struct read_ctx read_ctxs[MAX_STAGES]; /* Read context */
  struct lookahead_entry *buf;           /* Buffer list */
  int push_frame_count; /* Number of frames that have been pushed in the queue*/
  uint8_t
      max_pre_frames; /* Maximum number of past frames allowed in the queue */
};
/*!\endcond */

/**\brief Initializes the lookahead stage
 *
 * The lookahead stage is a queue of frame buffers on which some analysis
 * may be done when buffers are enqueued.
 */
struct lookahead_ctx *av1_lookahead_init(
    unsigned int width, unsigned int height, unsigned int subsampling_x,
    unsigned int subsampling_y, int use_highbitdepth, unsigned int depth,
    const int border_in_pixels, int byte_alignment, int num_lap_buffers,
    bool is_all_intra, bool alloc_pyramid);

/**\brief Destroys the lookahead stage
 */
void av1_lookahead_destroy(struct lookahead_ctx *ctx);

/**\brief Check if lookahead buffer is full
 */
int av1_lookahead_full(const struct lookahead_ctx *ctx);

/**\brief Enqueue a source buffer
 *
 * This function will copy the source image into a new framebuffer with
 * the expected stride/border.
 *
 * \param[in] ctx               Pointer to the lookahead context
 * \param[in] src               Pointer to the image to enqueue
 * \param[in] ts_start          Timestamp for the start of this frame
 * \param[in] ts_end            Timestamp for the end of this frame
 * \param[in] use_highbitdepth  Tell if HBD is used
 * \param[in] alloc_pyramid     Whether to allocate a downsampling pyramid
 *                              for each frame buffer
 * \param[in] flags             Flags set on this frame
 */
int av1_lookahead_push(struct lookahead_ctx *ctx, const YV12_BUFFER_CONFIG *src,
                       int64_t ts_start, int64_t ts_end, int use_highbitdepth,
                       bool alloc_pyramid, aom_enc_frame_flags_t flags);

/**\brief Get the next source buffer to encode
 *
 * \param[in] ctx       Pointer to the lookahead context
 * \param[in] drain     Flag indicating the buffer should be drained
 *                      (return a buffer regardless of the current queue depth)
 * \param[in] stage     Encoder stage
 *
 * \retval Return NULL, if drain set and queue is empty, or if drain not set and
 * queue not of the configured depth.
 */
struct lookahead_entry *av1_lookahead_pop(struct lookahead_ctx *ctx, int drain,
                                          COMPRESSOR_STAGE stage);

/**\brief Get a future source buffer to encode
 *
 * \param[in] ctx       Pointer to the lookahead context
 * \param[in] index     Index of the frame to be returned, 0 == next frame
 * \param[in] stage     Encoder stage
 *
 * \retval Return NULL, if no buffer exists at the specified index
 */
struct lookahead_entry *av1_lookahead_peek(struct lookahead_ctx *ctx, int index,
                                           COMPRESSOR_STAGE stage);

/**\brief Get the number of frames currently in the lookahead queue
 */
unsigned int av1_lookahead_depth(struct lookahead_ctx *ctx,
                                 COMPRESSOR_STAGE stage);

/**\brief Get pop_sz value
 */
int av1_lookahead_pop_sz(struct lookahead_ctx *ctx, COMPRESSOR_STAGE stage);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_LOOKAHEAD_H_
