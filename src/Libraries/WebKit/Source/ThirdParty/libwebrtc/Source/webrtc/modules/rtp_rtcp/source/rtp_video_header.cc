/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 12, 2022.
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
#include "modules/rtp_rtcp/source/rtp_video_header.h"

namespace webrtc {

RTPVideoHeader::GenericDescriptorInfo::GenericDescriptorInfo() = default;
RTPVideoHeader::GenericDescriptorInfo::GenericDescriptorInfo(
    const GenericDescriptorInfo& other) = default;
RTPVideoHeader::GenericDescriptorInfo::~GenericDescriptorInfo() = default;

// static
RTPVideoHeader RTPVideoHeader::FromMetadata(
    const VideoFrameMetadata& metadata) {
  RTPVideoHeader rtp_video_header;
  rtp_video_header.SetFromMetadata(metadata);
  return rtp_video_header;
}

RTPVideoHeader::RTPVideoHeader() : video_timing() {}
RTPVideoHeader::RTPVideoHeader(const RTPVideoHeader& other) = default;
RTPVideoHeader::~RTPVideoHeader() = default;

VideoFrameMetadata RTPVideoHeader::GetAsMetadata() const {
  VideoFrameMetadata metadata;
  metadata.SetFrameType(frame_type);
  metadata.SetWidth(width);
  metadata.SetHeight(height);
  metadata.SetRotation(rotation);
  metadata.SetContentType(content_type);
  if (generic) {
    metadata.SetFrameId(generic->frame_id);
    metadata.SetSpatialIndex(generic->spatial_index);
    metadata.SetTemporalIndex(generic->temporal_index);
    metadata.SetFrameDependencies(generic->dependencies);
    metadata.SetDecodeTargetIndications(generic->decode_target_indications);
  }
  metadata.SetIsLastFrameInPicture(is_last_frame_in_picture);
  metadata.SetSimulcastIdx(simulcastIdx);
  metadata.SetCodec(codec);
  switch (codec) {
    case VideoCodecType::kVideoCodecVP8:
      metadata.SetRTPVideoHeaderCodecSpecifics(
          absl::get<RTPVideoHeaderVP8>(video_type_header));
      break;
    case VideoCodecType::kVideoCodecVP9:
      metadata.SetRTPVideoHeaderCodecSpecifics(
          absl::get<RTPVideoHeaderVP9>(video_type_header));
      break;
    case VideoCodecType::kVideoCodecH264:
      metadata.SetRTPVideoHeaderCodecSpecifics(
          absl::get<RTPVideoHeaderH264>(video_type_header));
      break;
    // These codec types do not have codec-specifics.
    case VideoCodecType::kVideoCodecH265:
    case VideoCodecType::kVideoCodecAV1:
    case VideoCodecType::kVideoCodecGeneric:
      break;
  }
  return metadata;
}

void RTPVideoHeader::SetFromMetadata(const VideoFrameMetadata& metadata) {
  frame_type = metadata.GetFrameType();
  width = metadata.GetWidth();
  height = metadata.GetHeight();
  rotation = metadata.GetRotation();
  content_type = metadata.GetContentType();
  if (!metadata.GetFrameId().has_value()) {
    generic = std::nullopt;
  } else {
    generic.emplace();
    generic->frame_id = metadata.GetFrameId().value();
    generic->spatial_index = metadata.GetSpatialIndex();
    generic->temporal_index = metadata.GetTemporalIndex();
    generic->dependencies.assign(metadata.GetFrameDependencies().begin(),
                                 metadata.GetFrameDependencies().end());
    generic->decode_target_indications.assign(
        metadata.GetDecodeTargetIndications().begin(),
        metadata.GetDecodeTargetIndications().end());
  }
  is_last_frame_in_picture = metadata.GetIsLastFrameInPicture();
  simulcastIdx = metadata.GetSimulcastIdx();
  codec = metadata.GetCodec();
  switch (codec) {
    case VideoCodecType::kVideoCodecVP8:
      video_type_header = absl::get<RTPVideoHeaderVP8>(
          metadata.GetRTPVideoHeaderCodecSpecifics());
      break;
    case VideoCodecType::kVideoCodecVP9:
      video_type_header = absl::get<RTPVideoHeaderVP9>(
          metadata.GetRTPVideoHeaderCodecSpecifics());
      break;
    case VideoCodecType::kVideoCodecH264:
      video_type_header = absl::get<RTPVideoHeaderH264>(
          metadata.GetRTPVideoHeaderCodecSpecifics());
      break;
    default:
      // Codec-specifics are not supported for this codec.
      break;
  }
}

}  // namespace webrtc
