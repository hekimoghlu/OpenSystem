/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 11, 2024.
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
#ifndef SRC_TARGETS_PARSER_H_
#define SRC_TARGETS_PARSER_H_

#include "src/byte_parser.h"
#include "src/int_parser.h"
#include "src/master_value_parser.h"
#include "webm/dom_types.h"
#include "webm/id.h"

namespace webm {

// Spec reference:
// http://matroska.org/technical/specs/index.html#Targets
// http://www.webmproject.org/docs/container/#Targets
class TargetsParser : public MasterValueParser<Targets> {
 public:
  TargetsParser()
      : MasterValueParser<Targets>(
            MakeChild<UnsignedIntParser>(Id::kTargetTypeValue,
                                         &Targets::type_value),
            MakeChild<StringParser>(Id::kTargetType, &Targets::type),
            MakeChild<UnsignedIntParser>(Id::kTagTrackUid,
                                         &Targets::track_uids)) {}
};

}  // namespace webm

#endif  // SRC_TARGETS_PARSER_H_
