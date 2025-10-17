/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 31, 2022.
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
#ifndef SRC_CLUSTER_PARSER_H_
#define SRC_CLUSTER_PARSER_H_

#include "src/block_group_parser.h"
#include "src/block_parser.h"
#include "src/int_parser.h"
#include "src/master_value_parser.h"
#include "webm/dom_types.h"
#include "webm/id.h"

namespace webm {

// Spec reference:
// http://matroska.org/technical/specs/index.html#Cluster
// http://www.webmproject.org/docs/container/#Cluster
class ClusterParser : public MasterValueParser<Cluster> {
 public:
  ClusterParser()
      : MasterValueParser<Cluster>(
            MakeChild<UnsignedIntParser>(Id::kTimecode, &Cluster::timecode),
            MakeChild<UnsignedIntParser>(Id::kPrevSize,
                                         &Cluster::previous_size),
            MakeChild<SimpleBlockParser>(Id::kSimpleBlock,
                                         &Cluster::simple_blocks)
                .UseAsStartEvent(),
            MakeChild<BlockGroupParser>(Id::kBlockGroup, &Cluster::block_groups)
                .UseAsStartEvent()) {}

 protected:
  Status OnParseStarted(Callback* callback, Action* action) override {
    return callback->OnClusterBegin(metadata(Id::kCluster), value(), action);
  }

  Status OnParseCompleted(Callback* callback) override {
    return callback->OnClusterEnd(metadata(Id::kCluster), value());
  }
};

}  // namespace webm

#endif  // SRC_CLUSTER_PARSER_H_
