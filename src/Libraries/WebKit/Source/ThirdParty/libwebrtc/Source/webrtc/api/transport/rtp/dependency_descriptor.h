/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 18, 2025.
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
#ifndef API_TRANSPORT_RTP_DEPENDENCY_DESCRIPTOR_H_
#define API_TRANSPORT_RTP_DEPENDENCY_DESCRIPTOR_H_

#include <stdint.h>

#include <initializer_list>
#include <memory>
#include <optional>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "api/video/render_resolution.h"

namespace webrtc {
// Structures to build and parse dependency descriptor as described in
// https://aomediacodec.github.io/av1-rtp-spec/#dependency-descriptor-rtp-header-extension

// Relationship of a frame to a Decode target.
enum class DecodeTargetIndication {
  kNotPresent = 0,   // DecodeTargetInfo symbol '-'
  kDiscardable = 1,  // DecodeTargetInfo symbol 'D'
  kSwitch = 2,       // DecodeTargetInfo symbol 'S'
  kRequired = 3      // DecodeTargetInfo symbol 'R'
};

struct FrameDependencyTemplate {
  // Setters are named briefly to chain them when building the template.
  FrameDependencyTemplate& S(int spatial_layer);
  FrameDependencyTemplate& T(int temporal_layer);
  FrameDependencyTemplate& Dtis(absl::string_view dtis);
  FrameDependencyTemplate& FrameDiffs(std::initializer_list<int> diffs);
  FrameDependencyTemplate& ChainDiffs(std::initializer_list<int> diffs);

  friend bool operator==(const FrameDependencyTemplate& lhs,
                         const FrameDependencyTemplate& rhs) {
    return lhs.spatial_id == rhs.spatial_id &&
           lhs.temporal_id == rhs.temporal_id &&
           lhs.decode_target_indications == rhs.decode_target_indications &&
           lhs.frame_diffs == rhs.frame_diffs &&
           lhs.chain_diffs == rhs.chain_diffs;
  }

  int spatial_id = 0;
  int temporal_id = 0;
  absl::InlinedVector<DecodeTargetIndication, 10> decode_target_indications;
  absl::InlinedVector<int, 4> frame_diffs;
  absl::InlinedVector<int, 4> chain_diffs;
};

struct FrameDependencyStructure {
  friend bool operator==(const FrameDependencyStructure& lhs,
                         const FrameDependencyStructure& rhs) {
    return lhs.num_decode_targets == rhs.num_decode_targets &&
           lhs.num_chains == rhs.num_chains &&
           lhs.decode_target_protected_by_chain ==
               rhs.decode_target_protected_by_chain &&
           lhs.resolutions == rhs.resolutions && lhs.templates == rhs.templates;
  }

  int structure_id = 0;
  int num_decode_targets = 0;
  int num_chains = 0;
  // If chains are used (num_chains > 0), maps decode target index into index of
  // the chain protecting that target.
  absl::InlinedVector<int, 10> decode_target_protected_by_chain;
  absl::InlinedVector<RenderResolution, 4> resolutions;
  std::vector<FrameDependencyTemplate> templates;
};

class DependencyDescriptorMandatory {
 public:
  void set_frame_number(int frame_number) { frame_number_ = frame_number; }
  int frame_number() const { return frame_number_; }

  void set_template_id(int template_id) { template_id_ = template_id; }
  int template_id() const { return template_id_; }

  void set_first_packet_in_frame(bool first) { first_packet_in_frame_ = first; }
  bool first_packet_in_frame() const { return first_packet_in_frame_; }

  void set_last_packet_in_frame(bool last) { last_packet_in_frame_ = last; }
  bool last_packet_in_frame() const { return last_packet_in_frame_; }

 private:
  int frame_number_;
  int template_id_;
  bool first_packet_in_frame_;
  bool last_packet_in_frame_;
};

struct DependencyDescriptor {
  static constexpr int kMaxSpatialIds = 4;
  static constexpr int kMaxTemporalIds = 8;
  static constexpr int kMaxDecodeTargets = 32;
  static constexpr int kMaxTemplates = 64;

  bool first_packet_in_frame = true;
  bool last_packet_in_frame = true;
  int frame_number = 0;
  FrameDependencyTemplate frame_dependencies;
  std::optional<RenderResolution> resolution;
  std::optional<uint32_t> active_decode_targets_bitmask;
  std::unique_ptr<FrameDependencyStructure> attached_structure;
};

// Below are implementation details.
namespace webrtc_impl {
absl::InlinedVector<DecodeTargetIndication, 10> StringToDecodeTargetIndications(
    absl::string_view indication_symbols);
}  // namespace webrtc_impl

inline FrameDependencyTemplate& FrameDependencyTemplate::S(int spatial_layer) {
  this->spatial_id = spatial_layer;
  return *this;
}
inline FrameDependencyTemplate& FrameDependencyTemplate::T(int temporal_layer) {
  this->temporal_id = temporal_layer;
  return *this;
}
inline FrameDependencyTemplate& FrameDependencyTemplate::Dtis(
    absl::string_view dtis) {
  this->decode_target_indications =
      webrtc_impl::StringToDecodeTargetIndications(dtis);
  return *this;
}
inline FrameDependencyTemplate& FrameDependencyTemplate::FrameDiffs(
    std::initializer_list<int> diffs) {
  this->frame_diffs.assign(diffs.begin(), diffs.end());
  return *this;
}
inline FrameDependencyTemplate& FrameDependencyTemplate::ChainDiffs(
    std::initializer_list<int> diffs) {
  this->chain_diffs.assign(diffs.begin(), diffs.end());
  return *this;
}

}  // namespace webrtc

#endif  // API_TRANSPORT_RTP_DEPENDENCY_DESCRIPTOR_H_
