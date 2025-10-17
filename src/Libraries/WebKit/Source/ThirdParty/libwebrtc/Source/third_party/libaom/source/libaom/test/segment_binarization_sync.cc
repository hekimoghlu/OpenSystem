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
#include "gtest/gtest.h"

#include "av1/common/seg_common.h"
#include "av1/decoder/decodemv.h"
#include "av1/encoder/bitstream.h"
#include "test/acm_random.h"

using libaom_test::ACMRandom;

namespace {

struct Segment {
  int id;
  int pred;
  int last_id;
};

Segment GenerateSegment(int seed) {
  ACMRandom rnd_(seed);

  Segment segment;
  const int last_segid = rnd_.PseudoUniform(MAX_SEGMENTS);
  segment.last_id = last_segid;
  segment.pred = rnd_.PseudoUniform(MAX_SEGMENTS);
  segment.id = rnd_.PseudoUniform(last_segid + 1);

  return segment;
}

// Try to reveal a mismatch between segment binarization and debinarization
TEST(SegmentBinarizationSync, SearchForBinarizationMismatch) {
  const int count_tests = 1000;
  const int seed_init = 4321;

  for (int i = 0; i < count_tests; ++i) {
    const Segment seg = GenerateSegment(seed_init + i);

    const int max_segid = seg.last_id + 1;
    const int seg_diff = av1_neg_interleave(seg.id, seg.pred, max_segid);
    const int decoded_segid =
        av1_neg_deinterleave(seg_diff, seg.pred, max_segid);

    ASSERT_EQ(decoded_segid, seg.id);
  }
}

}  // namespace
