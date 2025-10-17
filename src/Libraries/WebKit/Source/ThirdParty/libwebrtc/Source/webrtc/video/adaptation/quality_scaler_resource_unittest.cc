/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 21, 2024.
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

#include <memory>
#include <optional>

#include "api/task_queue/task_queue_base.h"
#include "api/video_codecs/video_encoder.h"
#include "call/adaptation/test/mock_resource_listener.h"
#include "rtc_base/thread.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {

using testing::_;
using testing::Eq;
using testing::StrictMock;

namespace {

class FakeDegradationPreferenceProvider : public DegradationPreferenceProvider {
 public:
  ~FakeDegradationPreferenceProvider() override = default;

  DegradationPreference degradation_preference() const override {
    return DegradationPreference::MAINTAIN_FRAMERATE;
  }
};

}  // namespace

class QualityScalerResourceTest : public ::testing::Test {
 public:
  QualityScalerResourceTest()
      : quality_scaler_resource_(QualityScalerResource::Create()) {
    quality_scaler_resource_->RegisterEncoderTaskQueue(
        TaskQueueBase::Current());
    quality_scaler_resource_->SetResourceListener(&fake_resource_listener_);
  }

  ~QualityScalerResourceTest() override {
    quality_scaler_resource_->SetResourceListener(nullptr);
  }

 protected:
  rtc::AutoThread main_thread_;
  StrictMock<MockResourceListener> fake_resource_listener_;
  FakeDegradationPreferenceProvider degradation_preference_provider_;
  rtc::scoped_refptr<QualityScalerResource> quality_scaler_resource_;
};

TEST_F(QualityScalerResourceTest, ReportQpHigh) {
  EXPECT_CALL(fake_resource_listener_,
              OnResourceUsageStateMeasured(Eq(quality_scaler_resource_),
                                           Eq(ResourceUsageState::kOveruse)));
  quality_scaler_resource_->OnReportQpUsageHigh();
}

TEST_F(QualityScalerResourceTest, ReportQpLow) {
  EXPECT_CALL(fake_resource_listener_,
              OnResourceUsageStateMeasured(Eq(quality_scaler_resource_),
                                           Eq(ResourceUsageState::kUnderuse)));
  quality_scaler_resource_->OnReportQpUsageLow();
}

}  // namespace webrtc
