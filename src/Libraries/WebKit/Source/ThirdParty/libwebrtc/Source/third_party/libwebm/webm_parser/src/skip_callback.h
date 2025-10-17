/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 22, 2025.
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
#ifndef SRC_SKIP_CALLBACK_H_
#define SRC_SKIP_CALLBACK_H_

#include "webm/callback.h"
#include "webm/dom_types.h"
#include "webm/reader.h"
#include "webm/status.h"

namespace webm {

// An implementation of Callback that skips all elements. Every method that
// yields an action will yield Action::kSkip, and Reader::Skip will be called
// if the callback ever needs to process data from the byte stream.
class SkipCallback : public Callback {
 public:
  Status OnElementBegin(const ElementMetadata& /* metadata */,
                        Action* action) override {
    *action = Action::kSkip;
    return Status(Status::kOkCompleted);
  }

  Status OnSegmentBegin(const ElementMetadata& /* metadata */,
                        Action* action) override {
    *action = Action::kSkip;
    return Status(Status::kOkCompleted);
  }

  Status OnClusterBegin(const ElementMetadata& /* metadata */,
                        const Cluster& /* cluster */, Action* action) override {
    *action = Action::kSkip;
    return Status(Status::kOkCompleted);
  }

  Status OnSimpleBlockBegin(const ElementMetadata& /* metadata */,
                            const SimpleBlock& /* simple_block */,
                            Action* action) override {
    *action = Action::kSkip;
    return Status(Status::kOkCompleted);
  }

  Status OnBlockGroupBegin(const ElementMetadata& /* metadata */,
                           Action* action) override {
    *action = Action::kSkip;
    return Status(Status::kOkCompleted);
  }

  Status OnBlockBegin(const ElementMetadata& /* metadata */,
                      const Block& /* block */, Action* action) override {
    *action = Action::kSkip;
    return Status(Status::kOkCompleted);
  }
};

}  // namespace webm

#endif  // SRC_SKIP_CALLBACK_H_
