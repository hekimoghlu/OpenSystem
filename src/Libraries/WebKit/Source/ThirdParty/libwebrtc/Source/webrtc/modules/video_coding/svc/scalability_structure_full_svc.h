/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 28, 2025.
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
#ifndef MODULES_VIDEO_CODING_SVC_SCALABILITY_STRUCTURE_FULL_SVC_H_
#define MODULES_VIDEO_CODING_SVC_SCALABILITY_STRUCTURE_FULL_SVC_H_

#include <bitset>
#include <vector>

#include "api/transport/rtp/dependency_descriptor.h"
#include "common_video/generic_frame_descriptor/generic_frame_info.h"
#include "modules/video_coding/svc/scalable_video_controller.h"

namespace webrtc {

class ScalabilityStructureFullSvc : public ScalableVideoController {
 public:
  struct ScalingFactor {
    int num = 1;
    int den = 2;
  };
  ScalabilityStructureFullSvc(int num_spatial_layers,
                              int num_temporal_layers,
                              ScalingFactor resolution_factor);
  ~ScalabilityStructureFullSvc() override;

  StreamLayersConfig StreamConfig() const override;

  std::vector<LayerFrameConfig> NextFrameConfig(bool restart) override;
  GenericFrameInfo OnEncodeDone(const LayerFrameConfig& config) override;
  void OnRatesUpdated(const VideoBitrateAllocation& bitrates) override;

 private:
  enum FramePattern {
    kNone,
    kKey,
    kDeltaT2A,
    kDeltaT1,
    kDeltaT2B,
    kDeltaT0,
  };
  static constexpr absl::string_view kFramePatternNames[] = {
      "None", "Key", "DeltaT2A", "DeltaT1", "DeltaT2B", "DeltaT0"};
  static constexpr int kMaxNumSpatialLayers = 3;
  static constexpr int kMaxNumTemporalLayers = 3;

  // Index of the buffer to store last frame for layer (`sid`, `tid`)
  int BufferIndex(int sid, int tid) const {
    return tid * num_spatial_layers_ + sid;
  }
  bool DecodeTargetIsActive(int sid, int tid) const {
    return active_decode_targets_[sid * num_temporal_layers_ + tid];
  }
  void SetDecodeTargetIsActive(int sid, int tid, bool value) {
    active_decode_targets_.set(sid * num_temporal_layers_ + tid, value);
  }
  FramePattern NextPattern() const;
  bool TemporalLayerIsActive(int tid) const;
  static DecodeTargetIndication Dti(int sid,
                                    int tid,
                                    const LayerFrameConfig& frame);

  const int num_spatial_layers_;
  const int num_temporal_layers_;
  const ScalingFactor resolution_factor_;

  FramePattern last_pattern_ = kNone;
  std::bitset<kMaxNumSpatialLayers> can_reference_t0_frame_for_spatial_id_ = 0;
  std::bitset<kMaxNumSpatialLayers> can_reference_t1_frame_for_spatial_id_ = 0;
  std::bitset<32> active_decode_targets_;
};

// T1       0   0
//         /   /   / ...
// T0     0---0---0--
// Time-> 0 1 2 3 4
class ScalabilityStructureL1T2 : public ScalabilityStructureFullSvc {
 public:
  explicit ScalabilityStructureL1T2(ScalingFactor resolution_factor = {})
      : ScalabilityStructureFullSvc(1, 2, resolution_factor) {}
  ~ScalabilityStructureL1T2() override = default;

  FrameDependencyStructure DependencyStructure() const override;
};

// T2       0   0   0   0
//          |  /    |  /
// T1       / 0     / 0  ...
//         |_/     |_/
// T0     0-------0------
// Time-> 0 1 2 3 4 5 6 7
class ScalabilityStructureL1T3 : public ScalabilityStructureFullSvc {
 public:
  explicit ScalabilityStructureL1T3(ScalingFactor resolution_factor = {})
      : ScalabilityStructureFullSvc(1, 3, resolution_factor) {}
  ~ScalabilityStructureL1T3() override = default;

  FrameDependencyStructure DependencyStructure() const override;
};

// S1  0--0--0-
//     |  |  | ...
// S0  0--0--0-
class ScalabilityStructureL2T1 : public ScalabilityStructureFullSvc {
 public:
  explicit ScalabilityStructureL2T1(ScalingFactor resolution_factor = {})
      : ScalabilityStructureFullSvc(2, 1, resolution_factor) {}
  ~ScalabilityStructureL2T1() override = default;

  FrameDependencyStructure DependencyStructure() const override;
};

// S1T1     0   0
//         /|  /|  /
// S1T0   0-+-0-+-0
//        | | | | | ...
// S0T1   | 0 | 0 |
//        |/  |/  |/
// S0T0   0---0---0--
// Time-> 0 1 2 3 4
class ScalabilityStructureL2T2 : public ScalabilityStructureFullSvc {
 public:
  explicit ScalabilityStructureL2T2(ScalingFactor resolution_factor = {})
      : ScalabilityStructureFullSvc(2, 2, resolution_factor) {}
  ~ScalabilityStructureL2T2() override = default;

  FrameDependencyStructure DependencyStructure() const override;
};

// S1T2      4    ,8
// S1T1    / |  6' |
// S1T0   2--+-'+--+-...
//        |  |  |  |
// S0T2   |  3  | ,7
// S0T1   | /   5'
// S0T0   1----'-----...
// Time-> 0  1  2  3
class ScalabilityStructureL2T3 : public ScalabilityStructureFullSvc {
 public:
  explicit ScalabilityStructureL2T3(ScalingFactor resolution_factor = {})
      : ScalabilityStructureFullSvc(2, 3, resolution_factor) {}
  ~ScalabilityStructureL2T3() override = default;

  FrameDependencyStructure DependencyStructure() const override;
};

// S2     0-0-0-
//        | | |
// S1     0-0-0-...
//        | | |
// S0     0-0-0-
// Time-> 0 1 2
class ScalabilityStructureL3T1 : public ScalabilityStructureFullSvc {
 public:
  explicit ScalabilityStructureL3T1(ScalingFactor resolution_factor = {})
      : ScalabilityStructureFullSvc(3, 1, resolution_factor) {}
  ~ScalabilityStructureL3T1() override = default;

  FrameDependencyStructure DependencyStructure() const override;
};

// https://www.w3.org/TR/webrtc-svc/#L3T2*
class ScalabilityStructureL3T2 : public ScalabilityStructureFullSvc {
 public:
  explicit ScalabilityStructureL3T2(ScalingFactor resolution_factor = {})
      : ScalabilityStructureFullSvc(3, 2, resolution_factor) {}
  ~ScalabilityStructureL3T2() override = default;

  FrameDependencyStructure DependencyStructure() const override;
};

// https://www.w3.org/TR/webrtc-svc/#L3T3*
class ScalabilityStructureL3T3 : public ScalabilityStructureFullSvc {
 public:
  explicit ScalabilityStructureL3T3(ScalingFactor resolution_factor = {})
      : ScalabilityStructureFullSvc(3, 3, resolution_factor) {}
  ~ScalabilityStructureL3T3() override = default;

  FrameDependencyStructure DependencyStructure() const override;
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_SVC_SCALABILITY_STRUCTURE_FULL_SVC_H_
