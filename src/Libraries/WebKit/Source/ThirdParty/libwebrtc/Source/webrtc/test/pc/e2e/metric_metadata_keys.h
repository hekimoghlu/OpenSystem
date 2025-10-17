/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 9, 2025.
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
#ifndef TEST_PC_E2E_METRIC_METADATA_KEYS_H_
#define TEST_PC_E2E_METRIC_METADATA_KEYS_H_

#include <string>

namespace webrtc {
namespace webrtc_pc_e2e {

// All metadata fields are present only if applicable for particular metric.
class MetricMetadataKey {
 public:
  // Represents on peer with whom the metric is associated.
  static constexpr char kPeerMetadataKey[] = "peer";
  // Represents sender of the media stream.
  static constexpr char kSenderMetadataKey[] = "sender";
  // Represents receiver of the media stream.
  static constexpr char kReceiverMetadataKey[] = "receiver";
  // Represents name of the audio stream.
  static constexpr char kAudioStreamMetadataKey[] = "audio_stream";
  // Represents name of the video stream.
  static constexpr char kVideoStreamMetadataKey[] = "video_stream";
  // Represents name of the sync group to which stream belongs.
  static constexpr char kPeerSyncGroupMetadataKey[] = "peer_sync_group";
  // Represents the test name (without any peer and stream data appended to it
  // as it currently happens with the webrtc.test_metrics.Metric.test_case
  // field). This metadata is temporary and it will be removed once this
  // information is moved to webrtc.test_metrics.Metric.test_case.
  // TODO(bugs.webrtc.org/14757): Remove kExperimentalTestNameMetadataKey.
  static constexpr char kExperimentalTestNameMetadataKey[] =
      "experimental_test_name";
  // Represents index of a video spatial layer to which metric belongs.
  static constexpr char kSpatialLayerMetadataKey[] = "spatial_layer";

 private:
  MetricMetadataKey() = default;
};

// All metadata fields are presented only if applicable for particular metric.
class SampleMetadataKey {
 public:
  // Represents a frame ID with which data point is associated.
  static constexpr char kFrameIdMetadataKey[] = "frame_id";

 private:
  SampleMetadataKey() = default;
};

}  // namespace webrtc_pc_e2e
}  // namespace webrtc

#endif  // TEST_PC_E2E_METRIC_METADATA_KEYS_H_
