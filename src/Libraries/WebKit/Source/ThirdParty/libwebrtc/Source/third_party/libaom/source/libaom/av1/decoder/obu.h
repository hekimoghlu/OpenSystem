/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 30, 2022.
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
#ifndef AOM_AV1_DECODER_OBU_H_
#define AOM_AV1_DECODER_OBU_H_

#include "aom/aom_codec.h"
#include "av1/decoder/decoder.h"

// Try to decode one frame from a buffer.
// Returns 1 if we decoded a frame,
//         0 if we didn't decode a frame but that's okay
//           (eg, if there was a frame but we skipped it),
//     or -1 on error
int aom_decode_frame_from_obus(struct AV1Decoder *pbi, const uint8_t *data,
                               const uint8_t *data_end,
                               const uint8_t **p_data_end);

aom_codec_err_t aom_get_num_layers_from_operating_point_idc(
    int operating_point_idc, unsigned int *number_spatial_layers,
    unsigned int *number_temporal_layers);

#endif  // AOM_AV1_DECODER_OBU_H_
