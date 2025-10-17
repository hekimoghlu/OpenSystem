/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 17, 2025.
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
#ifndef AOM_AV1_DECODER_DTHREAD_H_
#define AOM_AV1_DECODER_DTHREAD_H_

#include "config/aom_config.h"

#include "aom/internal/aom_codec_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

struct AV1Common;
struct AV1Decoder;
struct ThreadData;

typedef struct DecWorkerData {
  struct ThreadData *td;
  const uint8_t *data_end;
  struct aom_internal_error_info error_info;
} DecWorkerData;

// WorkerData for the FrameWorker thread. It contains all the information of
// the worker and decode structures for decoding a frame.
typedef struct FrameWorkerData {
  struct AV1Decoder *pbi;
  const uint8_t *data;
  const uint8_t *data_end;
  size_t data_size;
  void *user_priv;
  int received_frame;
  int frame_context_ready;  // Current frame's context is ready to read.
  int frame_decoded;        // Finished decoding current frame.
} FrameWorkerData;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_DECODER_DTHREAD_H_
