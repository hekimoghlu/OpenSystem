/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 19, 2022.
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
#ifndef AOM_COMMON_OBUDEC_H_
#define AOM_COMMON_OBUDEC_H_

#include "common/tools_common.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ObuDecInputContext {
  struct AvxInputContext *avx_ctx;
  uint8_t *buffer;
  size_t buffer_capacity;
  size_t bytes_buffered;
  int is_annexb;
};

// Returns 1 when file data starts (if Annex B stream, after reading the
// size of the OBU) with what appears to be a Temporal Delimiter
// OBU as defined by Section 5 of the AV1 bitstream specification.
int file_is_obu(struct ObuDecInputContext *obu_ctx);

// Reads one Temporal Unit from the input file. Returns 0 when a TU is
// successfully read, 1 when end of file is reached, and less than 0 when an
// error occurs. Stores TU data in 'buffer'. Reallocs buffer to match TU size,
// returns buffer capacity via 'buffer_size', and returns size of buffered data
// via 'bytes_read'.
int obudec_read_temporal_unit(struct ObuDecInputContext *obu_ctx,
                              uint8_t **buffer, size_t *bytes_read,
                              size_t *buffer_size);

void obudec_free(struct ObuDecInputContext *obu_ctx);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // AOM_COMMON_OBUDEC_H_
