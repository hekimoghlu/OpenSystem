/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 24, 2024.
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
#ifndef RTC_TOOLS_RTC_EVENT_LOG_VISUALIZER_ANALYZE_AUDIO_H_
#define RTC_TOOLS_RTC_EVENT_LOG_VISUALIZER_ANALYZE_AUDIO_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "api/function_view.h"
#include "api/neteq/neteq.h"
#include "logging/rtc_event_log/rtc_event_log_parser.h"
#include "modules/audio_coding/neteq/tools/neteq_stats_getter.h"
#include "rtc_tools/rtc_event_log_visualizer/analyzer_common.h"
#include "rtc_tools/rtc_event_log_visualizer/plot_base.h"

namespace webrtc {

void CreateAudioEncoderTargetBitrateGraph(const ParsedRtcEventLog& parsed_log,
                                          const AnalyzerConfig& config,
                                          Plot* plot);
void CreateAudioEncoderFrameLengthGraph(const ParsedRtcEventLog& parsed_log,
                                        const AnalyzerConfig& config,
                                        Plot* plot);
void CreateAudioEncoderPacketLossGraph(const ParsedRtcEventLog& parsed_log,
                                       const AnalyzerConfig& config,
                                       Plot* plot);
void CreateAudioEncoderEnableFecGraph(const ParsedRtcEventLog& parsed_log,
                                      const AnalyzerConfig& config,
                                      Plot* plot);
void CreateAudioEncoderEnableDtxGraph(const ParsedRtcEventLog& parsed_log,
                                      const AnalyzerConfig& config,
                                      Plot* plot);
void CreateAudioEncoderNumChannelsGraph(const ParsedRtcEventLog& parsed_log,
                                        const AnalyzerConfig& config,
                                        Plot* plot);

using NetEqStatsGetterMap =
    std::map<uint32_t, std::unique_ptr<test::NetEqStatsGetter>>;
NetEqStatsGetterMap SimulateNetEq(const ParsedRtcEventLog& parsed_log,
                                  const AnalyzerConfig& config,
                                  const std::string& replacement_file_name,
                                  int file_sample_rate_hz);

void CreateAudioJitterBufferGraph(const ParsedRtcEventLog& parsed_log,
                                  const AnalyzerConfig& config,
                                  uint32_t ssrc,
                                  const test::NetEqStatsGetter* stats_getter,
                                  Plot* plot);
void CreateNetEqNetworkStatsGraph(
    const ParsedRtcEventLog& parsed_log,
    const AnalyzerConfig& config,
    const NetEqStatsGetterMap& neteq_stats_getters,
    rtc::FunctionView<float(const NetEqNetworkStatistics&)> stats_extractor,
    const std::string& plot_name,
    Plot* plot);
void CreateNetEqLifetimeStatsGraph(
    const ParsedRtcEventLog& parsed_log,
    const AnalyzerConfig& config,
    const NetEqStatsGetterMap& neteq_stats_getters,
    rtc::FunctionView<float(const NetEqLifetimeStatistics&)> stats_extractor,
    const std::string& plot_name,
    Plot* plot);

}  // namespace webrtc

#endif  // RTC_TOOLS_RTC_EVENT_LOG_VISUALIZER_ANALYZE_AUDIO_H_
