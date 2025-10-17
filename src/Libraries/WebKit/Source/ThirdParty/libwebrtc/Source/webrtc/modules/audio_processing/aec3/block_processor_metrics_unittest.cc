/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 6, 2025.
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
#include "modules/audio_processing/aec3/block_processor_metrics.h"

#include "modules/audio_processing/aec3/aec3_common.h"
#include "test/gtest.h"

namespace webrtc {

// Verify the general functionality of BlockProcessorMetrics.
TEST(BlockProcessorMetrics, NormalUsage) {
  BlockProcessorMetrics metrics;

  for (int j = 0; j < 3; ++j) {
    for (int k = 0; k < kMetricsReportingIntervalBlocks - 1; ++k) {
      metrics.UpdateRender(false);
      metrics.UpdateRender(false);
      metrics.UpdateCapture(false);
      EXPECT_FALSE(metrics.MetricsReported());
    }
    metrics.UpdateCapture(false);
    EXPECT_TRUE(metrics.MetricsReported());
  }
}

}  // namespace webrtc
