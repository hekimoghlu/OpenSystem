/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 18, 2021.
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
/*!\defgroup vp8_decoder WebM VP8/VP9 Decoder
 * \ingroup vp8
 *
 * @{
 */
/*!\file
 * \brief Provides definitions for using VP8 or VP9 within the vpx Decoder
 *        interface.
 */
#ifndef VPX_VPX_VP8DX_H_
#define VPX_VPX_VP8DX_H_

#ifdef __cplusplus
extern "C" {
#endif

/* Include controls common to both the encoder and decoder */
#include "./vp8.h"

/*!\name Algorithm interface for VP8
 *
 * This interface provides the capability to decode VP8 streams.
 * @{
 */

/*!\brief A single instance of the VP8 decoder.
 *\deprecated This access mechanism is provided for backwards compatibility;
 * prefer vpx_codec_vp8_dx().
 */
extern vpx_codec_iface_t vpx_codec_vp8_dx_algo;

/*!\brief The interface to the VP8 decoder.
 */
extern vpx_codec_iface_t *vpx_codec_vp8_dx(void);
/*!@} - end algorithm interface member group*/

/*!\name Algorithm interface for VP9
 *
 * This interface provides the capability to decode VP9 streams.
 * @{
 */

/*!\brief A single instance of the VP9 decoder.
 *\deprecated This access mechanism is provided for backwards compatibility;
 * prefer vpx_codec_vp9_dx().
 */
extern vpx_codec_iface_t vpx_codec_vp9_dx_algo;

/*!\brief The interface to the VP9 decoder.
 */
extern vpx_codec_iface_t *vpx_codec_vp9_dx(void);
/*!@} - end algorithm interface member group*/

/*!\enum vp8_dec_control_id
 * \brief VP8 decoder control functions
 *
 * This set of macros define the control functions available for the VP8
 * decoder interface.
 *
 * \sa #vpx_codec_control
 */
enum vp8_dec_control_id {
  /** control function to get info on which reference frames were updated
   *  by the last decode
   */
  VP8D_GET_LAST_REF_UPDATES = VP8_DECODER_CTRL_ID_START,

  /** check if the indicated frame is corrupted */
  VP8D_GET_FRAME_CORRUPTED,

  /** control function to get info on which reference frames were used
   *  by the last decode
   */
  VP8D_GET_LAST_REF_USED,

  /** decryption function to decrypt encoded buffer data immediately
   * before decoding. Takes a vpx_decrypt_init, which contains
   * a callback function and opaque context pointer.
   */
  VPXD_SET_DECRYPTOR,
  VP8D_SET_DECRYPTOR = VPXD_SET_DECRYPTOR,

  /** control function to get the dimensions that the current frame is decoded
   * at. This may be different to the intended display size for the frame as
   * specified in the wrapper or frame header (see VP9D_GET_DISPLAY_SIZE). */
  VP9D_GET_FRAME_SIZE,

  /** control function to get the current frame's intended display dimensions
   * (as specified in the wrapper or frame header). This may be different to
   * the decoded dimensions of this frame (see VP9D_GET_FRAME_SIZE). */
  VP9D_GET_DISPLAY_SIZE,

  /** control function to get the bit depth of the stream. */
  VP9D_GET_BIT_DEPTH,

  /** control function to set the byte alignment of the planes in the reference
   * buffers. Valid values are power of 2, from 32 to 1024. A value of 0 sets
   * legacy alignment. I.e. Y plane is aligned to 32 bytes, U plane directly
   * follows Y plane, and V plane directly follows U plane. Default value is 0.
   */
  VP9_SET_BYTE_ALIGNMENT,

  /** control function to invert the decoding order to from right to left. The
   * function is used in a test to confirm the decoding independence of tile
   * columns. The function may be used in application where this order
   * of decoding is desired.
   *
   * TODO(yaowu): Rework the unit test that uses this control, and in a future
   *              release, this test-only control shall be removed.
   */
  VP9_INVERT_TILE_DECODE_ORDER,

  /** control function to set the skip loop filter flag. Valid values are
   * integers. The decoder will skip the loop filter when its value is set to
   * nonzero. If the loop filter is skipped the decoder may accumulate decode
   * artifacts. The default value is 0.
   */
  VP9_SET_SKIP_LOOP_FILTER,

  /** control function to decode SVC stream up to the x spatial layers,
   * where x is passed in through the control, and is 0 for base layer.
   */
  VP9_DECODE_SVC_SPATIAL_LAYER,

  /*!\brief Codec control function to get last decoded frame quantizer.
   *
   * Return value uses internal quantizer scale defined by the codec.
   *
   * Supported in codecs: VP8, VP9
   */
  VPXD_GET_LAST_QUANTIZER,

  /*!\brief Codec control function to set row level multi-threading.
   *
   * 0 : off, 1 : on
   *
   * Supported in codecs: VP9
   */
  VP9D_SET_ROW_MT,

  /*!\brief Codec control function to set loopfilter optimization.
   *
   * 0 : off, Loop filter is done after all tiles have been decoded
   * 1 : on, Loop filter is done immediately after decode without
   *     waiting for all threads to sync.
   *
   * Supported in codecs: VP9
   */
  VP9D_SET_LOOP_FILTER_OPT,

  VP8_DECODER_CTRL_ID_MAX
};

/** Decrypt n bytes of data from input -> output, using the decrypt_state
 *  passed in VPXD_SET_DECRYPTOR.
 */
typedef void (*vpx_decrypt_cb)(void *decrypt_state, const unsigned char *input,
                               unsigned char *output, int count);

/*!\brief Structure to hold decryption state
 *
 * Defines a structure to hold the decryption state and access function.
 */
typedef struct vpx_decrypt_init {
  /*! Decrypt callback. */
  vpx_decrypt_cb decrypt_cb;

  /*! Decryption state. */
  void *decrypt_state;
} vpx_decrypt_init;

/*!\cond */
/*!\brief VP8 decoder control function parameter type
 *
 * Defines the data types that VP8D control functions take. Note that
 * additional common controls are defined in vp8.h
 *
 */

VPX_CTRL_USE_TYPE(VP8D_GET_LAST_REF_UPDATES, int *)
#define VPX_CTRL_VP8D_GET_LAST_REF_UPDATES
VPX_CTRL_USE_TYPE(VP8D_GET_FRAME_CORRUPTED, int *)
#define VPX_CTRL_VP8D_GET_FRAME_CORRUPTED
VPX_CTRL_USE_TYPE(VP8D_GET_LAST_REF_USED, int *)
#define VPX_CTRL_VP8D_GET_LAST_REF_USED
VPX_CTRL_USE_TYPE(VPXD_SET_DECRYPTOR, vpx_decrypt_init *)
#define VPX_CTRL_VPXD_SET_DECRYPTOR
VPX_CTRL_USE_TYPE(VP8D_SET_DECRYPTOR, vpx_decrypt_init *)
#define VPX_CTRL_VP8D_SET_DECRYPTOR
VPX_CTRL_USE_TYPE(VP9D_GET_FRAME_SIZE, int *)
#define VPX_CTRL_VP9D_GET_FRAME_SIZE
VPX_CTRL_USE_TYPE(VP9D_GET_DISPLAY_SIZE, int *)
#define VPX_CTRL_VP9D_GET_DISPLAY_SIZE
VPX_CTRL_USE_TYPE(VP9D_GET_BIT_DEPTH, unsigned int *)
#define VPX_CTRL_VP9D_GET_BIT_DEPTH
VPX_CTRL_USE_TYPE(VP9_SET_BYTE_ALIGNMENT, int)
#define VPX_CTRL_VP9_SET_BYTE_ALIGNMENT
VPX_CTRL_USE_TYPE(VP9_INVERT_TILE_DECODE_ORDER, int)
#define VPX_CTRL_VP9_INVERT_TILE_DECODE_ORDER
VPX_CTRL_USE_TYPE(VP9_SET_SKIP_LOOP_FILTER, int)
#define VPX_CTRL_VP9_SET_SKIP_LOOP_FILTER
VPX_CTRL_USE_TYPE(VP9_DECODE_SVC_SPATIAL_LAYER, int)
#define VPX_CTRL_VP9_DECODE_SVC_SPATIAL_LAYER
VPX_CTRL_USE_TYPE(VPXD_GET_LAST_QUANTIZER, int *)
#define VPX_CTRL_VPXD_GET_LAST_QUANTIZER
VPX_CTRL_USE_TYPE(VP9D_SET_ROW_MT, int)
#define VPX_CTRL_VP9_DECODE_SET_ROW_MT
VPX_CTRL_USE_TYPE(VP9D_SET_LOOP_FILTER_OPT, int)
#define VPX_CTRL_VP9_SET_LOOP_FILTER_OPT

/*!\endcond */
/*! @} - end defgroup vp8_decoder */

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPX_VP8DX_H_
