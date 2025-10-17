/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 6, 2024.
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
#ifndef AOM_AV1_DECODER_DECODEFRAME_H_
#define AOM_AV1_DECODER_DECODEFRAME_H_

#ifdef __cplusplus
extern "C" {
#endif

struct AV1Decoder;
struct aom_read_bit_buffer;
struct ThreadData;

// Reads the middle part of the sequence header OBU (from
// frame_width_bits_minus_1 to enable_restoration) into seq_params.
// Reports errors by calling rb->error_handler() or aom_internal_error().
void av1_read_sequence_header(AV1_COMMON *cm, struct aom_read_bit_buffer *rb,
                              SequenceHeader *seq_params);

BITSTREAM_PROFILE av1_read_profile(struct aom_read_bit_buffer *rb);

// Returns 0 on success. Sets pbi->common.error.error_code and returns -1 on
// failure.
int av1_check_trailing_bits(struct AV1Decoder *pbi,
                            struct aom_read_bit_buffer *rb);

// On success, returns the frame header size. On failure, calls
// aom_internal_error and does not return.
uint32_t av1_decode_frame_headers_and_setup(struct AV1Decoder *pbi,
                                            struct aom_read_bit_buffer *rb,
                                            int trailing_bits_present);

void av1_decode_tg_tiles_and_wrapup(struct AV1Decoder *pbi, const uint8_t *data,
                                    const uint8_t *data_end,
                                    const uint8_t **p_data_end, int start_tile,
                                    int end_tile, int initialize_flag);

// Implements the color_config() function in the spec. Reports errors by
// calling rb->error_handler() or aom_internal_error().
void av1_read_color_config(struct aom_read_bit_buffer *rb,
                           int allow_lowbitdepth, SequenceHeader *seq_params,
                           struct aom_internal_error_info *error_info);

// Implements the timing_info() function in the spec. Reports errors by calling
// rb->error_handler() or aom_internal_error().
void av1_read_timing_info_header(aom_timing_info_t *timing_info,
                                 struct aom_internal_error_info *error,
                                 struct aom_read_bit_buffer *rb);

// Implements the decoder_model_info() function in the spec. Reports errors by
// calling rb->error_handler().
void av1_read_decoder_model_info(aom_dec_model_info_t *decoder_model_info,
                                 struct aom_read_bit_buffer *rb);

// Implements the operating_parameters_info() function in the spec. Reports
// errors by calling rb->error_handler().
void av1_read_op_parameters_info(aom_dec_model_op_parameters_t *op_params,
                                 int buffer_delay_length,
                                 struct aom_read_bit_buffer *rb);

struct aom_read_bit_buffer *av1_init_read_bit_buffer(
    struct AV1Decoder *pbi, struct aom_read_bit_buffer *rb, const uint8_t *data,
    const uint8_t *data_end);

void av1_free_mc_tmp_buf(struct ThreadData *thread_data);

void av1_set_single_tile_decoding_mode(AV1_COMMON *const cm);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_DECODER_DECODEFRAME_H_
