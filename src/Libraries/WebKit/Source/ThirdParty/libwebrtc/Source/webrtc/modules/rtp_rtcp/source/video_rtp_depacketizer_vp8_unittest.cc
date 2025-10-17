/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 23, 2022.
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
#include "modules/rtp_rtcp/source/video_rtp_depacketizer_vp8.h"

#include "api/array_view.h"
#include "modules/rtp_rtcp/source/rtp_format_vp8.h"
#include "modules/rtp_rtcp/source/rtp_packet_to_send.h"
#include "rtc_base/copy_on_write_buffer.h"
#include "test/gmock.h"
#include "test/gtest.h"

// VP8 payload descriptor
// https://datatracker.ietf.org/doc/html/rfc7741#section-4.2
//
//       0 1 2 3 4 5 6 7
//      +-+-+-+-+-+-+-+-+
//      |X|R|N|S|R| PID | (REQUIRED)
//      +-+-+-+-+-+-+-+-+
// X:   |I|L|T|K| RSV   | (OPTIONAL)
//      +-+-+-+-+-+-+-+-+
// I:   |M| PictureID   | (OPTIONAL)
//      +-+-+-+-+-+-+-+-+
//      |   PictureID   |
//      +-+-+-+-+-+-+-+-+
// L:   |   TL0PICIDX   | (OPTIONAL)
//      +-+-+-+-+-+-+-+-+
// T/K: |TID|Y| KEYIDX  | (OPTIONAL)
//      +-+-+-+-+-+-+-+-+
//
// VP8 payload header. Considered part of the actual payload, sent to decoder.
// https://datatracker.ietf.org/doc/html/rfc7741#section-4.3
//
//       0 1 2 3 4 5 6 7
//      +-+-+-+-+-+-+-+-+
//      |Size0|H| VER |P|
//      +-+-+-+-+-+-+-+-+
//      :      ...      :
//      +-+-+-+-+-+-+-+-+

namespace webrtc {
namespace {

TEST(VideoRtpDepacketizerVp8Test, BasicHeader) {
  uint8_t packet[4] = {0};
  packet[0] = 0b0001'0100;  // S = 1, partition ID = 4.
  packet[1] = 0x01;         // P frame.

  RTPVideoHeader video_header;
  int offset = VideoRtpDepacketizerVp8::ParseRtpPayload(packet, &video_header);

  EXPECT_EQ(offset, 1);
  EXPECT_EQ(video_header.frame_type, VideoFrameType::kVideoFrameDelta);
  EXPECT_EQ(video_header.codec, kVideoCodecVP8);
  const auto& vp8_header =
      absl::get<RTPVideoHeaderVP8>(video_header.video_type_header);
  EXPECT_FALSE(vp8_header.nonReference);
  EXPECT_TRUE(vp8_header.beginningOfPartition);
  EXPECT_EQ(vp8_header.partitionId, 4);
  EXPECT_EQ(vp8_header.pictureId, kNoPictureId);
  EXPECT_EQ(vp8_header.tl0PicIdx, kNoTl0PicIdx);
  EXPECT_EQ(vp8_header.temporalIdx, kNoTemporalIdx);
  EXPECT_EQ(vp8_header.keyIdx, kNoKeyIdx);
}

TEST(VideoRtpDepacketizerVp8Test, OneBytePictureID) {
  const uint8_t kPictureId = 17;
  uint8_t packet[10] = {0};
  packet[0] = 0b1010'0000;
  packet[1] = 0b1000'0000;
  packet[2] = kPictureId;

  RTPVideoHeader video_header;
  int offset = VideoRtpDepacketizerVp8::ParseRtpPayload(packet, &video_header);

  EXPECT_EQ(offset, 3);
  const auto& vp8_header =
      absl::get<RTPVideoHeaderVP8>(video_header.video_type_header);
  EXPECT_EQ(vp8_header.pictureId, kPictureId);
}

TEST(VideoRtpDepacketizerVp8Test, TwoBytePictureID) {
  const uint16_t kPictureId = 0x1234;
  uint8_t packet[10] = {0};
  packet[0] = 0b1010'0000;
  packet[1] = 0b1000'0000;
  packet[2] = 0x80 | 0x12;
  packet[3] = 0x34;

  RTPVideoHeader video_header;
  int offset = VideoRtpDepacketizerVp8::ParseRtpPayload(packet, &video_header);

  EXPECT_EQ(offset, 4);
  const auto& vp8_header =
      absl::get<RTPVideoHeaderVP8>(video_header.video_type_header);
  EXPECT_EQ(vp8_header.pictureId, kPictureId);
}

TEST(VideoRtpDepacketizerVp8Test, Tl0PicIdx) {
  const uint8_t kTl0PicIdx = 17;
  uint8_t packet[13] = {0};
  packet[0] = 0b1000'0000;
  packet[1] = 0b0100'0000;
  packet[2] = kTl0PicIdx;

  RTPVideoHeader video_header;
  int offset = VideoRtpDepacketizerVp8::ParseRtpPayload(packet, &video_header);

  EXPECT_EQ(offset, 3);
  const auto& vp8_header =
      absl::get<RTPVideoHeaderVP8>(video_header.video_type_header);
  EXPECT_EQ(vp8_header.tl0PicIdx, kTl0PicIdx);
}

TEST(VideoRtpDepacketizerVp8Test, TIDAndLayerSync) {
  uint8_t packet[10] = {0};
  packet[0] = 0b1000'0000;
  packet[1] = 0b0010'0000;
  packet[2] = 0b10'0'00000;  // TID(2) + LayerSync(false)

  RTPVideoHeader video_header;
  int offset = VideoRtpDepacketizerVp8::ParseRtpPayload(packet, &video_header);

  EXPECT_EQ(offset, 3);
  const auto& vp8_header =
      absl::get<RTPVideoHeaderVP8>(video_header.video_type_header);
  EXPECT_EQ(vp8_header.temporalIdx, 2);
  EXPECT_FALSE(vp8_header.layerSync);
}

TEST(VideoRtpDepacketizerVp8Test, KeyIdx) {
  const uint8_t kKeyIdx = 17;
  uint8_t packet[10] = {0};
  packet[0] = 0b1000'0000;
  packet[1] = 0b0001'0000;
  packet[2] = kKeyIdx;

  RTPVideoHeader video_header;
  int offset = VideoRtpDepacketizerVp8::ParseRtpPayload(packet, &video_header);

  EXPECT_EQ(offset, 3);
  const auto& vp8_header =
      absl::get<RTPVideoHeaderVP8>(video_header.video_type_header);
  EXPECT_EQ(vp8_header.keyIdx, kKeyIdx);
}

TEST(VideoRtpDepacketizerVp8Test, MultipleExtensions) {
  uint8_t packet[10] = {0};
  packet[0] = 0b1010'0110;  // X and N bit set, partition ID = 6
  packet[1] = 0b1111'0000;
  packet[2] = 0x80 | 0x12;  // PictureID, high 7 bits.
  packet[3] = 0x34;         // PictureID, low 8 bits.
  packet[4] = 42;           // Tl0PicIdx.
  packet[5] = 0b01'1'10001;

  RTPVideoHeader video_header;
  int offset = VideoRtpDepacketizerVp8::ParseRtpPayload(packet, &video_header);

  EXPECT_EQ(offset, 6);
  const auto& vp8_header =
      absl::get<RTPVideoHeaderVP8>(video_header.video_type_header);
  EXPECT_TRUE(vp8_header.nonReference);
  EXPECT_EQ(vp8_header.partitionId, 0b0110);
  EXPECT_EQ(vp8_header.pictureId, 0x1234);
  EXPECT_EQ(vp8_header.tl0PicIdx, 42);
  EXPECT_EQ(vp8_header.temporalIdx, 1);
  EXPECT_TRUE(vp8_header.layerSync);
  EXPECT_EQ(vp8_header.keyIdx, 0b10001);
}

TEST(VideoRtpDepacketizerVp8Test, TooShortHeader) {
  uint8_t packet[4] = {0};
  packet[0] = 0b1000'0000;
  packet[1] = 0b1111'0000;  // All extensions are enabled...
  packet[2] = 0x80 | 17;    // ... but only 2 bytes PictureID is provided.
  packet[3] = 17;           // PictureID, low 8 bits.

  RTPVideoHeader unused;
  EXPECT_EQ(VideoRtpDepacketizerVp8::ParseRtpPayload(packet, &unused), 0);
}

TEST(VideoRtpDepacketizerVp8Test, WithPacketizer) {
  uint8_t data[10] = {0};
  RtpPacketToSend packet(/*extenions=*/nullptr);
  RTPVideoHeaderVP8 input_header;
  input_header.nonReference = true;
  input_header.pictureId = 300;
  input_header.temporalIdx = 1;
  input_header.layerSync = false;
  input_header.tl0PicIdx = kNoTl0PicIdx;  // Disable.
  input_header.keyIdx = 31;
  RtpPacketizerVp8 packetizer(data, /*limits=*/{}, input_header);
  EXPECT_EQ(packetizer.NumPackets(), 1u);
  ASSERT_TRUE(packetizer.NextPacket(&packet));

  VideoRtpDepacketizerVp8 depacketizer;
  std::optional<VideoRtpDepacketizer::ParsedRtpPayload> parsed =
      depacketizer.Parse(packet.PayloadBuffer());
  ASSERT_TRUE(parsed);

  EXPECT_EQ(parsed->video_header.codec, kVideoCodecVP8);
  const auto& vp8_header =
      absl::get<RTPVideoHeaderVP8>(parsed->video_header.video_type_header);
  EXPECT_EQ(vp8_header.nonReference, input_header.nonReference);
  EXPECT_EQ(vp8_header.pictureId, input_header.pictureId);
  EXPECT_EQ(vp8_header.tl0PicIdx, input_header.tl0PicIdx);
  EXPECT_EQ(vp8_header.temporalIdx, input_header.temporalIdx);
  EXPECT_EQ(vp8_header.layerSync, input_header.layerSync);
  EXPECT_EQ(vp8_header.keyIdx, input_header.keyIdx);
}

TEST(VideoRtpDepacketizerVp8Test, ReferencesInputCopyOnWriteBuffer) {
  constexpr size_t kHeaderSize = 5;
  uint8_t packet[16] = {0};
  packet[0] = 0b1000'0000;
  packet[1] = 0b1111'0000;  // with all extensions,
  packet[2] = 15;           // and one-byte picture id.

  rtc::CopyOnWriteBuffer rtp_payload(packet);
  VideoRtpDepacketizerVp8 depacketizer;
  std::optional<VideoRtpDepacketizer::ParsedRtpPayload> parsed =
      depacketizer.Parse(rtp_payload);
  ASSERT_TRUE(parsed);

  EXPECT_EQ(parsed->video_payload.size(), rtp_payload.size() - kHeaderSize);
  // Compare pointers to check there was no copy on write buffer unsharing.
  EXPECT_EQ(parsed->video_payload.cdata(), rtp_payload.cdata() + kHeaderSize);
}

TEST(VideoRtpDepacketizerVp8Test, FailsOnEmptyPayload) {
  rtc::ArrayView<const uint8_t> empty;
  RTPVideoHeader video_header;
  EXPECT_EQ(VideoRtpDepacketizerVp8::ParseRtpPayload(empty, &video_header), 0);
}

}  // namespace
}  // namespace webrtc
