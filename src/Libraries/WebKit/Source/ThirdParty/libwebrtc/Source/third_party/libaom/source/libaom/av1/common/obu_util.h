/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 27, 2024.
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
#ifndef AOM_AV1_COMMON_OBU_UTIL_H_
#define AOM_AV1_COMMON_OBU_UTIL_H_

#include "aom/aom_codec.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  size_t size;  // Size (1 or 2 bytes) of the OBU header (including the
                // optional OBU extension header) in the bitstream.
  OBU_TYPE type;
  int has_size_field;
  int has_extension;  // Whether the optional OBU extension header is present.
  // The following fields come from the OBU extension header. They are set to 0
  // if has_extension is false.
  int temporal_layer_id;
  int spatial_layer_id;
} ObuHeader;

aom_codec_err_t aom_read_obu_header(uint8_t *buffer, size_t buffer_length,
                                    size_t *consumed, ObuHeader *header,
                                    int is_annexb);

aom_codec_err_t aom_read_obu_header_and_size(const uint8_t *data,
                                             size_t bytes_available,
                                             int is_annexb,
                                             ObuHeader *obu_header,
                                             size_t *const payload_size,
                                             size_t *const bytes_read);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_OBU_UTIL_H_
