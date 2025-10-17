/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 5, 2024.
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
#include "video/adaptation/quality_scaler_resource.h"

#include <utility>

#include "api/field_trials_view.h"
#include "rtc_base/checks.h"
#include "rtc_base/experiments/balanced_degradation_settings.h"
#include "rtc_base/time_utils.h"

namespace webrtc {

// static
rtc::scoped_refptr<QualityScalerResource> QualityScalerResource::Create() {
  return rtc::make_ref_counted<QualityScalerResource>();
}

QualityScalerResource::QualityScalerResource()
    : VideoStreamEncoderResource("QualityScalerResource"),
      quality_scaler_(nullptr) {}

QualityScalerResource::~QualityScalerResource() {
  RTC_DCHECK(!quality_scaler_);
}

bool QualityScalerResource::is_started() const {
  RTC_DCHECK_RUN_ON(encoder_queue());
  return quality_scaler_.get();
}

void QualityScalerResource::StartCheckForOveruse(
    VideoEncoder::QpThresholds qp_thresholds,
    const FieldTrialsView& field_trials) {
  RTC_DCHECK_RUN_ON(encoder_queue());
  RTC_DCHECK(!is_started());
  quality_scaler_ = std::make_unique<QualityScaler>(
      this, std::move(qp_thresholds), field_trials);
}

void QualityScalerResource::StopCheckForOveruse() {
  RTC_DCHECK_RUN_ON(encoder_queue());
  RTC_DCHECK(is_started());
  // Ensure we have no pending callbacks. This makes it safe to destroy the
  // QualityScaler and even task queues with tasks in-flight.
  quality_scaler_.reset();
}

void QualityScalerResource::SetQpThresholds(
    VideoEncoder::QpThresholds qp_thresholds) {
  RTC_DCHECK_RUN_ON(encoder_queue());
  RTC_DCHECK(is_started());
  quality_scaler_->SetQpThresholds(std::move(qp_thresholds));
}

void QualityScalerResource::OnEncodeCompleted(const EncodedImage& encoded_image,
                                              int64_t time_sent_in_us) {
  RTC_DCHECK_RUN_ON(encoder_queue());
  if (quality_scaler_ && encoded_image.qp_ >= 0) {
    quality_scaler_->ReportQp(encoded_image.qp_, time_sent_in_us);
  }
}

void QualityScalerResource::OnFrameDropped(
    EncodedImageCallback::DropReason reason) {
  RTC_DCHECK_RUN_ON(encoder_queue());
  if (!quality_scaler_)
    return;
  switch (reason) {
    case EncodedImageCallback::DropReason::kDroppedByMediaOptimizations:
      quality_scaler_->ReportDroppedFrameByMediaOpt();
      break;
    case EncodedImageCallback::DropReason::kDroppedByEncoder:
      quality_scaler_->ReportDroppedFrameByEncoder();
      break;
  }
}

void QualityScalerResource::OnReportQpUsageHigh() {
  OnResourceUsageStateMeasured(ResourceUsageState::kOveruse);
}

void QualityScalerResource::OnReportQpUsageLow() {
  OnResourceUsageStateMeasured(ResourceUsageState::kUnderuse);
}

}  // namespace webrtc
