/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 27, 2025.
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
#ifndef API_TEST_METRICS_METRICS_EXPORTER_H_
#define API_TEST_METRICS_METRICS_EXPORTER_H_

#include "api/array_view.h"
#include "api/test/metrics/metric.h"

namespace webrtc {
namespace test {

// Exports metrics in the requested format.
class MetricsExporter {
 public:
  virtual ~MetricsExporter() = default;

  // Exports specified metrics in a format that depends on the implementation.
  // Returns true if export succeeded, false otherwise.
  virtual bool Export(rtc::ArrayView<const Metric> metrics) = 0;
};

}  // namespace test
}  // namespace webrtc

#endif  // API_TEST_METRICS_METRICS_EXPORTER_H_
