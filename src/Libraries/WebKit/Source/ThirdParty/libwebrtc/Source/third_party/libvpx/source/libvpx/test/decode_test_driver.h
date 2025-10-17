/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 12, 2023.
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
#ifndef VPX_TEST_DECODE_TEST_DRIVER_H_
#define VPX_TEST_DECODE_TEST_DRIVER_H_
#include <cstring>
#include "gtest/gtest.h"
#include "./vpx_config.h"
#include "vpx/vpx_decoder.h"

namespace libvpx_test {

class CodecFactory;
class CompressedVideoSource;

// Provides an object to handle decoding output
class DxDataIterator {
 public:
  explicit DxDataIterator(vpx_codec_ctx_t *decoder)
      : decoder_(decoder), iter_(nullptr) {}

  const vpx_image_t *Next() { return vpx_codec_get_frame(decoder_, &iter_); }

 private:
  vpx_codec_ctx_t *decoder_;
  vpx_codec_iter_t iter_;
};

// Provides a simplified interface to manage one video decoding.
// Similar to Encoder class, the exact services should be added
// as more tests are added.
class Decoder {
 public:
  explicit Decoder(vpx_codec_dec_cfg_t cfg)
      : cfg_(cfg), flags_(0), init_done_(false) {
    memset(&decoder_, 0, sizeof(decoder_));
  }

  Decoder(vpx_codec_dec_cfg_t cfg, const vpx_codec_flags_t flag)
      : cfg_(cfg), flags_(flag), init_done_(false) {
    memset(&decoder_, 0, sizeof(decoder_));
  }

  virtual ~Decoder() { vpx_codec_destroy(&decoder_); }

  vpx_codec_err_t PeekStream(const uint8_t *cxdata, size_t size,
                             vpx_codec_stream_info_t *stream_info);

  vpx_codec_err_t DecodeFrame(const uint8_t *cxdata, size_t size);

  vpx_codec_err_t DecodeFrame(const uint8_t *cxdata, size_t size,
                              void *user_priv);

  DxDataIterator GetDxData() { return DxDataIterator(&decoder_); }

  void Control(int ctrl_id, int arg) { Control(ctrl_id, arg, VPX_CODEC_OK); }

  void Control(int ctrl_id, const void *arg) {
    InitOnce();
    const vpx_codec_err_t res = vpx_codec_control_(&decoder_, ctrl_id, arg);
    ASSERT_EQ(VPX_CODEC_OK, res) << DecodeError();
  }

  void Control(int ctrl_id, int arg, vpx_codec_err_t expected_value) {
    InitOnce();
    const vpx_codec_err_t res = vpx_codec_control_(&decoder_, ctrl_id, arg);
    ASSERT_EQ(expected_value, res) << DecodeError();
  }

  const char *DecodeError() {
    const char *detail = vpx_codec_error_detail(&decoder_);
    return detail ? detail : vpx_codec_error(&decoder_);
  }

  // Passes the external frame buffer information to libvpx.
  vpx_codec_err_t SetFrameBufferFunctions(
      vpx_get_frame_buffer_cb_fn_t cb_get,
      vpx_release_frame_buffer_cb_fn_t cb_release, void *user_priv) {
    InitOnce();
    return vpx_codec_set_frame_buffer_functions(&decoder_, cb_get, cb_release,
                                                user_priv);
  }

  const char *GetDecoderName() const {
    return vpx_codec_iface_name(CodecInterface());
  }

  bool IsVP8() const;

  vpx_codec_ctx_t *GetDecoder() { return &decoder_; }

 protected:
  virtual vpx_codec_iface_t *CodecInterface() const = 0;

  void InitOnce() {
    if (!init_done_) {
      const vpx_codec_err_t res =
          vpx_codec_dec_init(&decoder_, CodecInterface(), &cfg_, flags_);
      ASSERT_EQ(VPX_CODEC_OK, res) << DecodeError();
      init_done_ = true;
    }
  }

  vpx_codec_ctx_t decoder_;
  vpx_codec_dec_cfg_t cfg_;
  vpx_codec_flags_t flags_;
  bool init_done_;
};

// Common test functionality for all Decoder tests.
class DecoderTest {
 public:
  // Main decoding loop
  virtual void RunLoop(CompressedVideoSource *video);
  virtual void RunLoop(CompressedVideoSource *video,
                       const vpx_codec_dec_cfg_t &dec_cfg);

  virtual void set_cfg(const vpx_codec_dec_cfg_t &dec_cfg);
  virtual void set_flags(const vpx_codec_flags_t flags);

  // Hook to be called before decompressing every frame.
  virtual void PreDecodeFrameHook(const CompressedVideoSource & /*video*/,
                                  Decoder * /*decoder*/) {}

  // Hook to be called to handle decode result. Return true to continue.
  virtual bool HandleDecodeResult(const vpx_codec_err_t res_dec,
                                  const CompressedVideoSource & /*video*/,
                                  Decoder *decoder) {
    EXPECT_EQ(VPX_CODEC_OK, res_dec) << decoder->DecodeError();
    return VPX_CODEC_OK == res_dec;
  }

  // Hook to be called on every decompressed frame.
  virtual void DecompressedFrameHook(const vpx_image_t & /*img*/,
                                     const unsigned int /*frame_number*/) {}

  // Hook to be called on peek result
  virtual void HandlePeekResult(Decoder *const decoder,
                                CompressedVideoSource *video,
                                const vpx_codec_err_t res_peek);

 protected:
  explicit DecoderTest(const CodecFactory *codec)
      : codec_(codec), cfg_(), flags_(0) {}

  virtual ~DecoderTest() {}

  const CodecFactory *codec_;
  vpx_codec_dec_cfg_t cfg_;
  vpx_codec_flags_t flags_;
};

}  // namespace libvpx_test

#endif  // VPX_TEST_DECODE_TEST_DRIVER_H_
