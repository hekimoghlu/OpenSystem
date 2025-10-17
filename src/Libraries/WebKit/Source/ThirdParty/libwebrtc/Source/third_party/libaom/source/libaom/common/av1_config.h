/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 15, 2025.
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
#ifndef AOM_COMMON_AV1_CONFIG_H_
#define AOM_COMMON_AV1_CONFIG_H_

#include "aom/aom_integer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Struct representing ISOBMFF/Matroska AV1 config. See:
// https://aomediacodec.github.io/av1-isobmff/#av1codecconfigurationbox-syntax
//
// The AV1 config has the following format:
//
// unsigned int (1) marker = 1;
// unsigned int (7) version = 1;
// unsigned int (3) seq_profile;
// unsigned int (5) seq_level_idx_0;
// unsigned int (1) seq_tier_0;
// unsigned int (1) high_bitdepth;
// unsigned int (1) twelve_bit;
// unsigned int (1) monochrome;
// unsigned int (1) chroma_subsampling_x;
// unsigned int (1) chroma_subsampling_y;
// unsigned int (2) chroma_sample_position;
// unsigned int (3) reserved = 0;
//
// unsigned int (1) initial_presentation_delay_present;
// if (initial_presentation_delay_present) {
//   unsigned int (4) initial_presentation_delay_minus_one;
// } else {
//   unsigned int (4) reserved = 0;
// }
//
// unsigned int (8)[] configOBUs;
//
// Note: get_av1config_from_obu() does not currently store 'configOBUs' data, so
// the field is omitted.
typedef struct _Av1Config {
  uint8_t marker;
  uint8_t version;
  uint8_t seq_profile;
  uint8_t seq_level_idx_0;
  uint8_t seq_tier_0;
  uint8_t high_bitdepth;
  uint8_t twelve_bit;
  uint8_t monochrome;
  uint8_t chroma_subsampling_x;
  uint8_t chroma_subsampling_y;
  uint8_t chroma_sample_position;
  uint8_t initial_presentation_delay_present;
  uint8_t initial_presentation_delay_minus_one;
} Av1Config;

// Attempts to parse a Sequence Header OBU and set the paramenters of 'config'.
// Returns 0 upon success, and -1 upon failure. 'buffer' can contain multiple
// OBUs, but the Sequence Header OBU must be the first OBU within the buffer.
int get_av1config_from_obu(const uint8_t *buffer, size_t length, int is_annexb,
                           Av1Config *config);

// Attempts to parse an AV1 config from 'buffer'. Returns 0 upon success.
// Returns -1 when 'buffer_length' is less than 4, when passed NULL pointers, or
// when parsing of 'buffer' fails.
int read_av1config(const uint8_t *buffer, size_t buffer_length,
                   size_t *bytes_read, Av1Config *config);

// Writes 'config' to 'buffer'. Returns 0 upon successful write to 'buffer'.
// Returns -1 when passed NULL pointers or when 'capacity' insufficient.
int write_av1config(const Av1Config *config, size_t capacity,
                    size_t *bytes_written, uint8_t *buffer);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // AOM_COMMON_AV1_CONFIG_H_
