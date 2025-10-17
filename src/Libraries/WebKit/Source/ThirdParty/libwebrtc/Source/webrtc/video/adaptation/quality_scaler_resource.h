/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 1, 2024.
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
#ifndef VIDEO_ADAPTATION_QUALITY_SCALER_RESOURCE_H_
#define VIDEO_ADAPTATION_QUALITY_SCALER_RESOURCE_H_

#include <memory>
#include <optional>
#include <queue>
#include <string>

#include "api/field_trials_view.h"
#include "api/scoped_refptr.h"
#include "api/video/video_adaptation_reason.h"
#include "api/video_codecs/video_encoder.h"
#include "call/adaptation/degradation_preference_provider.h"
#include "call/adaptation/resource_adaptation_processor_interface.h"
#include "modules/video_coding/utility/quality_scaler.h"
#include "video/adaptation/video_stream_encoder_resource.h"

namespace webrtc {

// Handles interaction with the QualityScaler.
class QualityScalerResource : public VideoStreamEncoderResource,
                              public QualityScalerQpUsageHandlerInterface {
 public:
  static rtc::scoped_refptr<QualityScalerResource> Create();

  QualityScalerResource();
  ~QualityScalerResource() override;

  bool is_started() const;

  void StartCheckForOveruse(VideoEncoder::QpThresholds qp_thresholds,
                            const FieldTrialsView& field_trials);
  void StopCheckForOveruse();
  void SetQpThresholds(VideoEncoder::QpThresholds qp_thresholds);
  void OnEncodeCompleted(const EncodedImage& encoded_image,
                         int64_t time_sent_in_us);
  void OnFrameDropped(EncodedImageCallback::DropReason reason);

  // QualityScalerQpUsageHandlerInterface implementation.
  void OnReportQpUsageHigh() override;
  void OnReportQpUsageLow() override;

 private:
  std::unique_ptr<QualityScaler> quality_scaler_
      RTC_GUARDED_BY(encoder_queue());
};

}  // namespace webrtc

#endif  // VIDEO_ADAPTATION_QUALITY_SCALER_RESOURCE_H_
