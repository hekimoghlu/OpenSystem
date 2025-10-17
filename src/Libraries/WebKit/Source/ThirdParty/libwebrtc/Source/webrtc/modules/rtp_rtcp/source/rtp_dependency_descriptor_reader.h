/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 18, 2022.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTP_DEPENDENCY_DESCRIPTOR_READER_H_
#define MODULES_RTP_RTCP_SOURCE_RTP_DEPENDENCY_DESCRIPTOR_READER_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "api/array_view.h"
#include "api/transport/rtp/dependency_descriptor.h"
#include "rtc_base/bitstream_reader.h"

namespace webrtc {
// Deserializes DependencyDescriptor rtp header extension.
class RtpDependencyDescriptorReader {
 public:
  // Parses the dependency descriptor.
  RtpDependencyDescriptorReader(rtc::ArrayView<const uint8_t> raw_data,
                                const FrameDependencyStructure* structure,
                                DependencyDescriptor* descriptor);
  RtpDependencyDescriptorReader(const RtpDependencyDescriptorReader&) = delete;
  RtpDependencyDescriptorReader& operator=(
      const RtpDependencyDescriptorReader&) = delete;

  // Returns true if parse was successful.
  bool ParseSuccessful() { return buffer_.Ok(); }

 private:
  // Functions to read template dependency structure.
  void ReadTemplateDependencyStructure();
  void ReadTemplateLayers();
  void ReadTemplateDtis();
  void ReadTemplateFdiffs();
  void ReadTemplateChains();
  void ReadResolutions();

  // Function to read details for the current frame.
  void ReadMandatoryFields();
  void ReadExtendedFields();
  void ReadFrameDependencyDefinition();

  void ReadFrameDtis();
  void ReadFrameFdiffs();
  void ReadFrameChains();

  // Output.
  DependencyDescriptor* const descriptor_;
  // Values that are needed while reading the descriptor, but can be discarded
  // when reading is complete.
  BitstreamReader buffer_;
  int frame_dependency_template_id_ = 0;
  bool active_decode_targets_present_flag_ = false;
  bool custom_dtis_flag_ = false;
  bool custom_fdiffs_flag_ = false;
  bool custom_chains_flag_ = false;
  const FrameDependencyStructure* structure_ = nullptr;
};

}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_SOURCE_RTP_DEPENDENCY_DESCRIPTOR_READER_H_
