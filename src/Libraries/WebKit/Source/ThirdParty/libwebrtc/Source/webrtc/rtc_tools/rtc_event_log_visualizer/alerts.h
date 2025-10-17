/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 20, 2023.
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
#ifndef RTC_TOOLS_RTC_EVENT_LOG_VISUALIZER_ALERTS_H_
#define RTC_TOOLS_RTC_EVENT_LOG_VISUALIZER_ALERTS_H_

#include <stdio.h>

#include <functional>
#include <map>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "logging/rtc_event_log/rtc_event_log_parser.h"
#include "rtc_tools/rtc_event_log_visualizer/analyzer_common.h"

namespace webrtc {

enum class TriageAlertType {
  kUnknown = 0,
  kIncomingRtpGap,
  kOutgoingRtpGap,
  kIncomingRtcpGap,
  kOutgoingRtcpGap,
  kIncomingSeqNumJump,
  kOutgoingSeqNumJump,
  kIncomingCaptureTimeJump,
  kOutgoingCaptureTimeJump,
  kOutgoingHighLoss,
  kLast,
};

struct TriageAlert {
  TriageAlertType type = TriageAlertType::kUnknown;
  int count = 0;
  float first_occurrence = -1;
  std::string explanation;
};

class TriageHelper {
 public:
  explicit TriageHelper(const AnalyzerConfig& config) : config_(config) {}

  TriageHelper(const TriageHelper&) = delete;
  TriageHelper& operator=(const TriageHelper&) = delete;

  void AnalyzeLog(const ParsedRtcEventLog& parsed_log);

  void AnalyzeStreamGaps(const ParsedRtcEventLog& parsed_log,
                         PacketDirection direction);
  void AnalyzeTransmissionGaps(const ParsedRtcEventLog& parsed_log,
                               PacketDirection direction);
  void Print(FILE* file);

  void ProcessAlerts(std::function<void(int, float, std::string)> f);

 private:
  AnalyzerConfig config_;
  std::map<TriageAlertType, TriageAlert> triage_alerts_;

  void Alert(TriageAlertType type,
             float time_seconds,
             absl::string_view explanation) {
    std::map<TriageAlertType, TriageAlert>::iterator it =
        triage_alerts_.find(type);

    if (it == triage_alerts_.end()) {
      TriageAlert alert;
      alert.type = type;
      alert.first_occurrence = time_seconds;
      alert.count = 1;
      alert.explanation = std::string(explanation);
      triage_alerts_.insert(std::make_pair(type, alert));
    } else {
      it->second.count += 1;
    }
  }
};

}  // namespace webrtc

#endif  // RTC_TOOLS_RTC_EVENT_LOG_VISUALIZER_ALERTS_H_
