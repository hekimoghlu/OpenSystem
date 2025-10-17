/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 7, 2022.
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
#include "modules/rtp_rtcp/source/rtp_video_layers_allocation_extension.h"

#include "api/video/video_layers_allocation.h"
#include "rtc_base/bit_buffer.h"
#include "rtc_base/buffer.h"
#include "test/gmock.h"

namespace webrtc {
namespace {

TEST(RtpVideoLayersAllocationExtension, WriteEmptyLayersAllocationReturnsTrue) {
  VideoLayersAllocation written_allocation;
  rtc::Buffer buffer(
      RtpVideoLayersAllocationExtension::ValueSize(written_allocation));
  EXPECT_TRUE(
      RtpVideoLayersAllocationExtension::Write(buffer, written_allocation));
}

TEST(RtpVideoLayersAllocationExtension,
     CanWriteAndParseLayersAllocationWithZeroSpatialLayers) {
  // We require the resolution_and_frame_rate_is_valid to be set to true in
  // order to send an "empty" allocation.
  VideoLayersAllocation written_allocation;
  written_allocation.resolution_and_frame_rate_is_valid = true;
  written_allocation.rtp_stream_index = 0;

  rtc::Buffer buffer(
      RtpVideoLayersAllocationExtension::ValueSize(written_allocation));
  EXPECT_TRUE(
      RtpVideoLayersAllocationExtension::Write(buffer, written_allocation));

  VideoLayersAllocation parsed_allocation;
  EXPECT_TRUE(
      RtpVideoLayersAllocationExtension::Parse(buffer, &parsed_allocation));
  EXPECT_EQ(written_allocation, parsed_allocation);
}

TEST(RtpVideoLayersAllocationExtension,
     CanWriteAndParse2SpatialWith2TemporalLayers) {
  VideoLayersAllocation written_allocation;
  written_allocation.rtp_stream_index = 1;
  written_allocation.active_spatial_layers = {
      {
          /*rtp_stream_index*/ 0,
          /*spatial_id*/ 0,
          /*target_bitrate_per_temporal_layer*/
          {DataRate::KilobitsPerSec(25), DataRate::KilobitsPerSec(50)},
          /*width*/ 0,
          /*height*/ 0,
          /*frame_rate_fps*/ 0,
      },
      {
          /*rtp_stream_index*/ 1,
          /*spatial_id*/ 0,
          /*target_bitrate_per_temporal_layer*/
          {DataRate::KilobitsPerSec(100), DataRate::KilobitsPerSec(200)},
          /*width*/ 0,
          /*height*/ 0,
          /*frame_rate_fps*/ 0,
      },
  };
  rtc::Buffer buffer(
      RtpVideoLayersAllocationExtension::ValueSize(written_allocation));
  EXPECT_TRUE(
      RtpVideoLayersAllocationExtension::Write(buffer, written_allocation));
  VideoLayersAllocation parsed_allocation;
  EXPECT_TRUE(
      RtpVideoLayersAllocationExtension::Parse(buffer, &parsed_allocation));
  EXPECT_EQ(written_allocation, parsed_allocation);
}

TEST(RtpVideoLayersAllocationExtension,
     CanWriteAndParseAllocationWithDifferentNumerOfSpatialLayers) {
  VideoLayersAllocation written_allocation;
  written_allocation.rtp_stream_index = 1;
  written_allocation.active_spatial_layers = {
      {/*rtp_stream_index*/ 0,
       /*spatial_id*/ 0,
       /*target_bitrate_per_temporal_layer*/ {DataRate::KilobitsPerSec(50)},
       /*width*/ 0,
       /*height*/ 0,
       /*frame_rate_fps*/ 0},
      {/*rtp_stream_index*/ 1,
       /*spatial_id*/ 0,
       /*target_bitrate_per_temporal_layer*/ {DataRate::KilobitsPerSec(100)},
       /*width*/ 0,
       /*height*/ 0,
       /*frame_rate_fps*/ 0},
      {/*rtp_stream_index*/ 1,
       /*spatial_id*/ 1,
       /*target_bitrate_per_temporal_layer*/ {DataRate::KilobitsPerSec(200)},
       /*width*/ 0,
       /*height*/ 0,
       /*frame_rate_fps*/ 0},
  };
  rtc::Buffer buffer(
      RtpVideoLayersAllocationExtension::ValueSize(written_allocation));
  EXPECT_TRUE(
      RtpVideoLayersAllocationExtension::Write(buffer, written_allocation));
  VideoLayersAllocation parsed_allocation;
  EXPECT_TRUE(
      RtpVideoLayersAllocationExtension::Parse(buffer, &parsed_allocation));
  EXPECT_EQ(written_allocation, parsed_allocation);
}

TEST(RtpVideoLayersAllocationExtension,
     CanWriteAndParseAllocationWithSkippedLowerSpatialLayer) {
  VideoLayersAllocation written_allocation;
  written_allocation.rtp_stream_index = 1;
  written_allocation.active_spatial_layers = {
      {/*rtp_stream_index*/ 0,
       /*spatial_id*/ 0,
       /*target_bitrate_per_temporal_layer*/ {DataRate::KilobitsPerSec(50)},
       /*width*/ 0,
       /*height*/ 0,
       /*frame_rate_fps*/ 0},
      {/*rtp_stream_index*/ 1,
       /*spatial_id*/ 1,
       /*target_bitrate_per_temporal_layer*/ {DataRate::KilobitsPerSec(200)},
       /*width*/ 0,
       /*height*/ 0,
       /*frame_rate_fps*/ 0},
  };
  rtc::Buffer buffer(
      RtpVideoLayersAllocationExtension::ValueSize(written_allocation));
  EXPECT_TRUE(
      RtpVideoLayersAllocationExtension::Write(buffer, written_allocation));
  VideoLayersAllocation parsed_allocation;
  EXPECT_TRUE(
      RtpVideoLayersAllocationExtension::Parse(buffer, &parsed_allocation));
  EXPECT_EQ(written_allocation, parsed_allocation);
}

TEST(RtpVideoLayersAllocationExtension,
     CanWriteAndParseAllocationWithSkippedRtpStreamIds) {
  VideoLayersAllocation written_allocation;
  written_allocation.rtp_stream_index = 2;
  written_allocation.active_spatial_layers = {
      {/*rtp_stream_index*/ 0,
       /*spatial_id*/ 0,
       /*target_bitrate_per_temporal_layer*/ {DataRate::KilobitsPerSec(50)},
       /*width*/ 0,
       /*height*/ 0,
       /*frame_rate_fps*/ 0},
      {/*rtp_stream_index*/ 2,
       /*spatial_id*/ 0,
       /*target_bitrate_per_temporal_layer*/ {DataRate::KilobitsPerSec(200)},
       /*width*/ 0,
       /*height*/ 0,
       /*frame_rate_fps*/ 0},
  };
  rtc::Buffer buffer(
      RtpVideoLayersAllocationExtension::ValueSize(written_allocation));
  EXPECT_TRUE(
      RtpVideoLayersAllocationExtension::Write(buffer, written_allocation));
  VideoLayersAllocation parsed_allocation;
  EXPECT_TRUE(
      RtpVideoLayersAllocationExtension::Parse(buffer, &parsed_allocation));
  EXPECT_EQ(written_allocation, parsed_allocation);
}

TEST(RtpVideoLayersAllocationExtension,
     CanWriteAndParseAllocationWithDifferentNumerOfTemporalLayers) {
  VideoLayersAllocation written_allocation;
  written_allocation.rtp_stream_index = 1;
  written_allocation.active_spatial_layers = {
      {
          /*rtp_stream_index*/ 0,
          /*spatial_id*/ 0,
          /*target_bitrate_per_temporal_layer*/
          {DataRate::KilobitsPerSec(25), DataRate::KilobitsPerSec(50)},
          /*width*/ 0,
          /*height*/ 0,
          /*frame_rate_fps*/ 0,
      },
      {
          /*rtp_stream_index*/ 1,
          /*spatial_id*/ 0,
          /*target_bitrate_per_temporal_layer*/ {DataRate::KilobitsPerSec(100)},
          /*width*/ 0,
          /*height*/ 0,
          /*frame_rate_fps*/ 0,
      },
  };
  rtc::Buffer buffer(
      RtpVideoLayersAllocationExtension::ValueSize(written_allocation));
  EXPECT_TRUE(
      RtpVideoLayersAllocationExtension::Write(buffer, written_allocation));
  VideoLayersAllocation parsed_allocation;
  EXPECT_TRUE(
      RtpVideoLayersAllocationExtension::Parse(buffer, &parsed_allocation));
  EXPECT_EQ(written_allocation, parsed_allocation);
}

TEST(RtpVideoLayersAllocationExtension,
     CanWriteAndParseAllocationWithResolution) {
  VideoLayersAllocation written_allocation;
  written_allocation.rtp_stream_index = 1;
  written_allocation.resolution_and_frame_rate_is_valid = true;
  written_allocation.active_spatial_layers = {
      {
          /*rtp_stream_index*/ 0,
          /*spatial_id*/ 0,
          /*target_bitrate_per_temporal_layer*/
          {DataRate::KilobitsPerSec(25), DataRate::KilobitsPerSec(50)},
          /*width*/ 320,
          /*height*/ 240,
          /*frame_rate_fps*/ 8,
      },
      {
          /*rtp_stream_index*/ 1,
          /*spatial_id*/ 1,
          /*target_bitrate_per_temporal_layer*/
          {DataRate::KilobitsPerSec(100), DataRate::KilobitsPerSec(200)},
          /*width*/ 640,
          /*height*/ 320,
          /*frame_rate_fps*/ 30,
      },
  };

  rtc::Buffer buffer(
      RtpVideoLayersAllocationExtension::ValueSize(written_allocation));
  EXPECT_TRUE(
      RtpVideoLayersAllocationExtension::Write(buffer, written_allocation));
  VideoLayersAllocation parsed_allocation;
  EXPECT_TRUE(
      RtpVideoLayersAllocationExtension::Parse(buffer, &parsed_allocation));
  EXPECT_EQ(written_allocation, parsed_allocation);
}

TEST(RtpVideoLayersAllocationExtension,
     WriteEmptyAllocationCanHaveAnyRtpStreamIndex) {
  VideoLayersAllocation written_allocation;
  written_allocation.rtp_stream_index = 1;
  rtc::Buffer buffer(
      RtpVideoLayersAllocationExtension::ValueSize(written_allocation));
  EXPECT_TRUE(
      RtpVideoLayersAllocationExtension::Write(buffer, written_allocation));
}

TEST(RtpVideoLayersAllocationExtension, DiscardsOverLargeDataRate) {
  constexpr uint8_t buffer[] = {0x4b, 0xf6, 0xff, 0xff, 0xff, 0xff, 0xff,
                                0xff, 0xcb, 0x78, 0xeb, 0x8d, 0xb5, 0x31};
  VideoLayersAllocation allocation;
  EXPECT_FALSE(RtpVideoLayersAllocationExtension::Parse(buffer, &allocation));
}

TEST(RtpVideoLayersAllocationExtension, DiscardsInvalidHeight) {
  VideoLayersAllocation written_allocation;
  written_allocation.rtp_stream_index = 0;
  written_allocation.resolution_and_frame_rate_is_valid = true;
  written_allocation.active_spatial_layers = {
      {
          /*rtp_stream_index*/ 0,
          /*spatial_id*/ 0,
          /*target_bitrate_per_temporal_layer*/
          {DataRate::KilobitsPerSec(25), DataRate::KilobitsPerSec(50)},
          /*width*/ 320,
          /*height*/ 240,
          /*frame_rate_fps*/ 8,
      },
  };
  rtc::Buffer buffer(
      RtpVideoLayersAllocationExtension::ValueSize(written_allocation));
  ASSERT_TRUE(
      RtpVideoLayersAllocationExtension::Write(buffer, written_allocation));

  // Modify the height to be invalid.
  buffer[buffer.size() - 3] = 0xff;
  buffer[buffer.size() - 2] = 0xff;
  VideoLayersAllocation allocation;
  EXPECT_FALSE(RtpVideoLayersAllocationExtension::Parse(buffer, &allocation));
}

}  // namespace
}  // namespace webrtc
