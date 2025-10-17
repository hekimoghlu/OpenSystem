/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 2, 2025.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTP_DEPENDENCY_DESCRIPTOR_WRITER_H_
#define MODULES_RTP_RTCP_SOURCE_RTP_DEPENDENCY_DESCRIPTOR_WRITER_H_

#include <bitset>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "api/array_view.h"
#include "api/transport/rtp/dependency_descriptor.h"
#include "rtc_base/bit_buffer.h"

namespace webrtc {
class RtpDependencyDescriptorWriter {
 public:
  // Assumes `structure` and `descriptor` are valid and
  // `descriptor` matches the `structure`.
  RtpDependencyDescriptorWriter(rtc::ArrayView<uint8_t> data,
                                const FrameDependencyStructure& structure,
                                std::bitset<32> active_chains,
                                const DependencyDescriptor& descriptor);

  // Serializes DependencyDescriptor rtp header extension.
  // Returns false if `data` is too small to serialize the `descriptor`.
  bool Write();

  // Returns minimum number of bits needed to serialize descriptor with respect
  // to the `structure`. Returns 0 if `descriptor` can't be serialized.
  int ValueSizeBits() const;

 private:
  // Used both as pointer to the template and as index in the templates vector.
  using TemplateIterator = std::vector<FrameDependencyTemplate>::const_iterator;
  struct TemplateMatch {
    TemplateIterator template_position;
    bool need_custom_dtis;
    bool need_custom_fdiffs;
    bool need_custom_chains;
    // Size in bits to store frame-specific details, i.e.
    // excluding mandatory fields and template dependency structure.
    int extra_size_bits;
  };
  int StructureSizeBits() const;
  TemplateMatch CalculateMatch(TemplateIterator frame_template) const;
  void FindBestTemplate();
  bool ShouldWriteActiveDecodeTargetsBitmask() const;
  bool HasExtendedFields() const;
  uint64_t TemplateId() const;

  void WriteBits(uint64_t val, size_t bit_count);
  void WriteNonSymmetric(uint32_t value, uint32_t num_values);

  // Functions to read template dependency structure.
  void WriteTemplateDependencyStructure();
  void WriteTemplateLayers();
  void WriteTemplateDtis();
  void WriteTemplateFdiffs();
  void WriteTemplateChains();
  void WriteResolutions();

  // Function to read details for the current frame.
  void WriteMandatoryFields();
  void WriteExtendedFields();
  void WriteFrameDependencyDefinition();

  void WriteFrameDtis();
  void WriteFrameFdiffs();
  void WriteFrameChains();

  bool build_failed_ = false;
  const DependencyDescriptor& descriptor_;
  const FrameDependencyStructure& structure_;
  std::bitset<32> active_chains_;
  rtc::BitBufferWriter bit_writer_;
  TemplateMatch best_template_;
};

}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_SOURCE_RTP_DEPENDENCY_DESCRIPTOR_WRITER_H_
