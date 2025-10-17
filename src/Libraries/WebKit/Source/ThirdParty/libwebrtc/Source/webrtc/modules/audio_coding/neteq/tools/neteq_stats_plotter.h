/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 7, 2025.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_TOOLS_NETEQ_STATS_PLOTTER_H_
#define MODULES_AUDIO_CODING_NETEQ_TOOLS_NETEQ_STATS_PLOTTER_H_

#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "modules/audio_coding/neteq/tools/neteq_delay_analyzer.h"
#include "modules/audio_coding/neteq/tools/neteq_stats_getter.h"
#include "modules/audio_coding/neteq/tools/neteq_test.h"

namespace webrtc {
namespace test {

class NetEqStatsPlotter : public NetEqSimulationEndedCallback {
 public:
  NetEqStatsPlotter(bool make_matlab_plot,
                    bool make_python_plot,
                    bool show_concealment_events,
                    absl::string_view base_file_name);

  void SimulationEnded(int64_t simulation_time_ms) override;

  NetEqStatsGetter* stats_getter() { return stats_getter_.get(); }

 private:
  std::unique_ptr<NetEqStatsGetter> stats_getter_;
  const bool make_matlab_plot_;
  const bool make_python_plot_;
  const bool show_concealment_events_;
  const std::string base_file_name_;
};

}  // namespace test
}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_NETEQ_TOOLS_NETEQ_STATS_PLOTTER_H_
