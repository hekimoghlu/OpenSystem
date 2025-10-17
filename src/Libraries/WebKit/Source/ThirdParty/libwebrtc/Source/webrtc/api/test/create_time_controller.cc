/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 12, 2021.
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
#include "api/test/create_time_controller.h"

#include <memory>
#include <utility>

#include "absl/base/nullability.h"
#include "api/enable_media_with_defaults.h"
#include "api/environment/environment.h"
#include "api/environment/environment_factory.h"
#include "api/peer_connection_interface.h"
#include "api/test/time_controller.h"
#include "api/units/timestamp.h"
#include "call/call.h"
#include "call/call_config.h"
#include "media/base/media_engine.h"
#include "pc/media_factory.h"
#include "rtc_base/checks.h"
#include "system_wrappers/include/clock.h"
#include "test/time_controller/external_time_controller.h"
#include "test/time_controller/simulated_time_controller.h"

namespace webrtc {

std::unique_ptr<TimeController> CreateTimeController(
    ControlledAlarmClock* alarm) {
  return std::make_unique<ExternalTimeController>(alarm);
}

std::unique_ptr<TimeController> CreateSimulatedTimeController() {
  return std::make_unique<GlobalSimulatedTimeController>(
      Timestamp::Seconds(10000));
}

void EnableMediaWithDefaultsAndTimeController(
    TimeController& time_controller,
    PeerConnectionFactoryDependencies& deps) {
  class TimeControllerBasedFactory : public MediaFactory {
   public:
    TimeControllerBasedFactory(
        absl::Nonnull<Clock*> clock,
        absl::Nonnull<std::unique_ptr<MediaFactory>> media_factory)
        : clock_(clock), media_factory_(std::move(media_factory)) {}

    std::unique_ptr<Call> CreateCall(CallConfig config) override {
      EnvironmentFactory env_factory(config.env);
      env_factory.Set(clock_);

      config.env = env_factory.Create();
      return media_factory_->CreateCall(std::move(config));
    }

    std::unique_ptr<cricket::MediaEngineInterface> CreateMediaEngine(
        const Environment& env,
        PeerConnectionFactoryDependencies& dependencies) override {
      return media_factory_->CreateMediaEngine(env, dependencies);
    }

   private:
    absl::Nonnull<Clock*> clock_;
    absl::Nonnull<std::unique_ptr<MediaFactory>> media_factory_;
  };

  EnableMediaWithDefaults(deps);
  RTC_CHECK(deps.media_factory);
  deps.media_factory = std::make_unique<TimeControllerBasedFactory>(
      time_controller.GetClock(), std::move(deps.media_factory));
}

}  // namespace webrtc
