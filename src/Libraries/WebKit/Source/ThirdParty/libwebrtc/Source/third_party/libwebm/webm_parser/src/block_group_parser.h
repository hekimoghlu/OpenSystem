/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 22, 2022.
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
#ifndef SRC_BLOCK_GROUP_PARSER_H_
#define SRC_BLOCK_GROUP_PARSER_H_

#include "src/block_additions_parser.h"
#include "src/block_parser.h"
#include "src/int_parser.h"
#include "src/master_value_parser.h"
#include "src/slices_parser.h"
#include "src/virtual_block_parser.h"
#include "webm/dom_types.h"
#include "webm/id.h"

namespace webm {

// Spec reference:
// http://matroska.org/technical/specs/index.html#BlockGroup
// http://www.webmproject.org/docs/container/#BlockGroup
class BlockGroupParser : public MasterValueParser<BlockGroup> {
 public:
  BlockGroupParser()
      : MasterValueParser<BlockGroup>(
            MakeChild<BlockParser>(Id::kBlock, &BlockGroup::block),
            MakeChild<VirtualBlockParser>(Id::kBlockVirtual,
                                          &BlockGroup::virtual_block),
            MakeChild<BlockAdditionsParser>(Id::kBlockAdditions,
                                            &BlockGroup::additions),
            MakeChild<UnsignedIntParser>(Id::kBlockDuration,
                                         &BlockGroup::duration),
            MakeChild<SignedIntParser>(Id::kReferenceBlock,
                                       &BlockGroup::references),
            MakeChild<SignedIntParser>(Id::kDiscardPadding,
                                       &BlockGroup::discard_padding),
            MakeChild<SlicesParser>(Id::kSlices, &BlockGroup::slices)) {}

  Status Feed(Callback* callback, Reader* reader,
              std::uint64_t* num_bytes_read) override {
    *num_bytes_read = 0;

    if (!parse_started_event_completed()) {
      Action action = Action::kRead;
      Status status = OnParseStarted(callback, &action);
      if (!status.completed_ok()) {
        return status;
      }

      set_parse_started_event_completed_with_action(action);
    }

    return MasterValueParser::Feed(callback, reader, num_bytes_read);
  }

 protected:
  Status OnParseStarted(Callback* callback, Action* action) override {
    return callback->OnBlockGroupBegin(metadata(Id::kBlockGroup), action);
  }

  Status OnParseCompleted(Callback* callback) override {
    return callback->OnBlockGroupEnd(metadata(Id::kBlockGroup), value());
  }
};

}  // namespace webm

#endif  // SRC_BLOCK_GROUP_PARSER_H_
