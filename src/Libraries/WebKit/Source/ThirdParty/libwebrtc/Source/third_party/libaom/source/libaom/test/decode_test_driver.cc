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
#include "gtest/gtest.h"

#include "test/codec_factory.h"
#include "test/decode_test_driver.h"
#include "test/register_state_check.h"
#include "test/video_source.h"

namespace libaom_test {

const char kAV1Name[] = "AOMedia Project AV1 Decoder";

aom_codec_err_t Decoder::PeekStream(const uint8_t *cxdata, size_t size,
                                    aom_codec_stream_info_t *stream_info) {
  return aom_codec_peek_stream_info(CodecInterface(), cxdata, size,
                                    stream_info);
}

aom_codec_err_t Decoder::DecodeFrame(const uint8_t *cxdata, size_t size) {
  return DecodeFrame(cxdata, size, nullptr);
}

aom_codec_err_t Decoder::DecodeFrame(const uint8_t *cxdata, size_t size,
                                     void *user_priv) {
  aom_codec_err_t res_dec;
  InitOnce();
  API_REGISTER_STATE_CHECK(
      res_dec = aom_codec_decode(&decoder_, cxdata, size, user_priv));
  return res_dec;
}

bool Decoder::IsAV1() const {
  const char *codec_name = GetDecoderName();
  return strncmp(kAV1Name, codec_name, sizeof(kAV1Name) - 1) == 0;
}

void DecoderTest::HandlePeekResult(Decoder *const /*decoder*/,
                                   CompressedVideoSource * /*video*/,
                                   const aom_codec_err_t res_peek) {
  /* The Av1 implementation of PeekStream returns an error only if the
   * data passed to it isn't a valid Av1 chunk. */
  ASSERT_EQ(AOM_CODEC_OK, res_peek)
      << "Peek return failed: " << aom_codec_err_to_string(res_peek);
}

void DecoderTest::RunLoop(CompressedVideoSource *video,
                          const aom_codec_dec_cfg_t &dec_cfg) {
  Decoder *const decoder = codec_->CreateDecoder(dec_cfg, flags_);
  ASSERT_NE(decoder, nullptr);
  bool end_of_file = false;
  bool peeked_stream = false;

  // Decode frames.
  for (video->Begin(); !::testing::Test::HasFailure() && !end_of_file;
       video->Next()) {
    PreDecodeFrameHook(*video, decoder);

    aom_codec_stream_info_t stream_info;
    stream_info.is_annexb = 0;

    if (video->cxdata() != nullptr) {
      if (!peeked_stream) {
        // TODO(yaowu): PeekStream returns error for non-sequence_header_obu,
        // therefore should only be tried once per sequence, this shall be fixed
        // once PeekStream is updated to properly operate on other obus.
        const aom_codec_err_t res_peek = decoder->PeekStream(
            video->cxdata(), video->frame_size(), &stream_info);
        HandlePeekResult(decoder, video, res_peek);
        ASSERT_FALSE(::testing::Test::HasFailure());
        peeked_stream = true;
      }

      aom_codec_err_t res_dec =
          decoder->DecodeFrame(video->cxdata(), video->frame_size());
      if (!HandleDecodeResult(res_dec, *video, decoder)) break;
    } else {
      // Signal end of the file to the decoder.
      const aom_codec_err_t res_dec = decoder->DecodeFrame(nullptr, 0);
      ASSERT_EQ(AOM_CODEC_OK, res_dec) << decoder->DecodeError();
      end_of_file = true;
    }

    DxDataIterator dec_iter = decoder->GetDxData();
    const aom_image_t *img = nullptr;

    // Get decompressed data
    while (!::testing::Test::HasFailure() && (img = dec_iter.Next()))
      DecompressedFrameHook(*img, video->frame_number());
  }
  delete decoder;
}

void DecoderTest::RunLoop(CompressedVideoSource *video) {
  aom_codec_dec_cfg_t dec_cfg = aom_codec_dec_cfg_t();
  RunLoop(video, dec_cfg);
}

void DecoderTest::set_cfg(const aom_codec_dec_cfg_t &dec_cfg) {
  memcpy(&cfg_, &dec_cfg, sizeof(cfg_));
}

void DecoderTest::set_flags(const aom_codec_flags_t flags) { flags_ = flags; }

}  // namespace libaom_test
