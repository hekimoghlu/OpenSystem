/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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
#ifndef PC_TEST_RTC_STATS_OBTAINER_H_
#define PC_TEST_RTC_STATS_OBTAINER_H_

#include "api/make_ref_counted.h"
#include "api/sequence_checker.h"
#include "api/stats/rtc_stats_collector_callback.h"
#include "api/stats/rtc_stats_report.h"
#include "rtc_base/gunit.h"

namespace webrtc {

class RTCStatsObtainer : public RTCStatsCollectorCallback {
 public:
  static rtc::scoped_refptr<RTCStatsObtainer> Create(
      rtc::scoped_refptr<const RTCStatsReport>* report_ptr = nullptr) {
    return rtc::make_ref_counted<RTCStatsObtainer>(report_ptr);
  }

  void OnStatsDelivered(
      const rtc::scoped_refptr<const RTCStatsReport>& report) override {
    EXPECT_TRUE(thread_checker_.IsCurrent());
    report_ = report;
    if (report_ptr_)
      *report_ptr_ = report_;
  }

  rtc::scoped_refptr<const RTCStatsReport> report() const {
    EXPECT_TRUE(thread_checker_.IsCurrent());
    return report_;
  }

 protected:
  explicit RTCStatsObtainer(
      rtc::scoped_refptr<const RTCStatsReport>* report_ptr)
      : report_ptr_(report_ptr) {}

 private:
  SequenceChecker thread_checker_;
  rtc::scoped_refptr<const RTCStatsReport> report_;
  rtc::scoped_refptr<const RTCStatsReport>* report_ptr_;
};

}  // namespace webrtc

#endif  // PC_TEST_RTC_STATS_OBTAINER_H_
