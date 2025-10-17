/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 22, 2024.
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
#include "pc/test/enable_fake_media.h"

#include <memory>
#include <utility>

#include "absl/base/nullability.h"
#include "api/environment/environment.h"
#include "api/peer_connection_interface.h"
#include "call/call.h"
#include "call/call_config.h"
#include "media/base/fake_media_engine.h"
#include "pc/media_factory.h"
#include "rtc_base/checks.h"

namespace webrtc {

using ::cricket::FakeMediaEngine;
using ::cricket::MediaEngineInterface;

void EnableFakeMedia(
    PeerConnectionFactoryDependencies& deps,
    absl::Nonnull<std::unique_ptr<FakeMediaEngine>> fake_media_engine) {
  class FakeMediaFactory : public MediaFactory {
   public:
    explicit FakeMediaFactory(
        absl::Nonnull<std::unique_ptr<FakeMediaEngine>> fake)
        : fake_(std::move(fake)) {}

    std::unique_ptr<Call> CreateCall(CallConfig config) override {
      return Call::Create(std::move(config));
    }

    std::unique_ptr<MediaEngineInterface> CreateMediaEngine(
        const Environment& /*env*/,
        PeerConnectionFactoryDependencies& /*dependencies*/) {
      RTC_CHECK(fake_ != nullptr)
          << "CreateMediaEngine can be called at most once.";
      return std::move(fake_);
    }

   private:
    absl::Nullable<std::unique_ptr<FakeMediaEngine>> fake_;
  };

  deps.media_factory =
      std::make_unique<FakeMediaFactory>(std::move(fake_media_engine));
}

void EnableFakeMedia(PeerConnectionFactoryDependencies& deps) {
  EnableFakeMedia(deps, std::make_unique<cricket::FakeMediaEngine>());
}

}  // namespace webrtc
