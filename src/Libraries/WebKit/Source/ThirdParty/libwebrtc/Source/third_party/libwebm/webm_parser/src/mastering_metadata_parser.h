/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 23, 2023.
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

// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_MASTERING_METADATA_PARSER_H_
#define SRC_MASTERING_METADATA_PARSER_H_

#include "src/float_parser.h"
#include "src/master_value_parser.h"
#include "webm/dom_types.h"
#include "webm/id.h"

namespace webm {

// Spec reference:
// http://matroska.org/technical/specs/index.html#MasteringMetadata
// http://www.webmproject.org/docs/container/#MasteringMetadata
class MasteringMetadataParser : public MasterValueParser<MasteringMetadata> {
 public:
  MasteringMetadataParser()
      : MasterValueParser<MasteringMetadata>(
            MakeChild<FloatParser>(
                Id::kPrimaryRChromaticityX,
                &MasteringMetadata::primary_r_chromaticity_x),
            MakeChild<FloatParser>(
                Id::kPrimaryRChromaticityY,
                &MasteringMetadata::primary_r_chromaticity_y),
            MakeChild<FloatParser>(
                Id::kPrimaryGChromaticityX,
                &MasteringMetadata::primary_g_chromaticity_x),
            MakeChild<FloatParser>(
                Id::kPrimaryGChromaticityY,
                &MasteringMetadata::primary_g_chromaticity_y),
            MakeChild<FloatParser>(
                Id::kPrimaryBChromaticityX,
                &MasteringMetadata::primary_b_chromaticity_x),
            MakeChild<FloatParser>(
                Id::kPrimaryBChromaticityY,
                &MasteringMetadata::primary_b_chromaticity_y),
            MakeChild<FloatParser>(
                Id::kWhitePointChromaticityX,
                &MasteringMetadata::white_point_chromaticity_x),
            MakeChild<FloatParser>(
                Id::kWhitePointChromaticityY,
                &MasteringMetadata::white_point_chromaticity_y),
            MakeChild<FloatParser>(Id::kLuminanceMax,
                                   &MasteringMetadata::luminance_max),
            MakeChild<FloatParser>(Id::kLuminanceMin,
                                   &MasteringMetadata::luminance_min)) {}
};

}  // namespace webm

#endif  // SRC_MASTERING_METADATA_PARSER_H_
