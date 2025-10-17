/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 9, 2023.
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

// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_OPERATORS_DECODER_HOST_FUSED_HOST_DECODER_RANDOM_CROP_H_
#define DALI_OPERATORS_DECODER_HOST_FUSED_HOST_DECODER_RANDOM_CROP_H_

#include <string>
#include <vector>

#include "dali/core/common.h"
#include "dali/operators/image/crop/random_crop_attr.h"
#include "dali/operators/decoder/host/host_decoder.h"

namespace dali {

class RandomCropGenerator;

class HostDecoderRandomCrop : public HostDecoder, public RandomCropAttr {
 public:
  explicit HostDecoderRandomCrop(const OpSpec &spec)
    : HostDecoder(spec)
    , RandomCropAttr(spec)
  {}

  inline ~HostDecoderRandomCrop() override = default;
  DISABLE_COPY_MOVE_ASSIGN(HostDecoderRandomCrop);

  void SaveState(OpCheckpoint &cpt, AccessOrder order) override;

  void RestoreState(const OpCheckpoint &cpt) override;

  std::string SerializeCheckpoint(const OpCheckpoint &cpt) const override;

  void DeserializeCheckpoint(OpCheckpoint &cpt, const std::string &data) const override;

 protected:
  inline CropWindowGenerator GetCropWindowGenerator(int data_idx) const override {
    return RandomCropAttr::GetCropWindowGenerator(data_idx);
  }
};

}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_HOST_FUSED_HOST_DECODER_RANDOM_CROP_H_
