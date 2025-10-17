/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 12, 2022.
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
#include <stdio.h>
#include <algorithm>
#include <cstdlib>

#include "test/bench.h"
#include "vpx_ports/vpx_timer.h"

void AbstractBench::RunNTimes(int n) {
  for (int r = 0; r < VPX_BENCH_ROBUST_ITER; r++) {
    vpx_usec_timer timer;
    vpx_usec_timer_start(&timer);
    for (int j = 0; j < n; ++j) {
      Run();
    }
    vpx_usec_timer_mark(&timer);
    times_[r] = static_cast<int>(vpx_usec_timer_elapsed(&timer));
  }
}

void AbstractBench::PrintMedian(const char *title) {
  std::sort(times_, times_ + VPX_BENCH_ROBUST_ITER);
  const int med = times_[VPX_BENCH_ROBUST_ITER >> 1];
  int sad = 0;
  for (int t = 0; t < VPX_BENCH_ROBUST_ITER; t++) {
    sad += abs(times_[t] - med);
  }
  printf("[%10s] %s %.1f ms ( Â±%.1f ms )\n", "BENCH ", title, med / 1000.0,
         sad / (VPX_BENCH_ROBUST_ITER * 1000.0));
}
