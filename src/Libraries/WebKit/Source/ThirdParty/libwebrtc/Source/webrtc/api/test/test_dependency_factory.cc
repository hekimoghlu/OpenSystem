/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 28, 2023.
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
#include "api/test/test_dependency_factory.h"

#include <memory>
#include <utility>

#include "api/test/video_quality_test_fixture.h"
#include "rtc_base/checks.h"
#include "rtc_base/platform_thread_types.h"

namespace webrtc {

namespace {
// This checks everything in this file gets called on the same thread. It's
// static because it needs to look at the static methods too.
bool IsValidTestDependencyFactoryThread() {
  const rtc::PlatformThreadRef main_thread = rtc::CurrentThreadRef();
  return rtc::IsThreadRefEqual(main_thread, rtc::CurrentThreadRef());
}
}  // namespace

std::unique_ptr<TestDependencyFactory> TestDependencyFactory::instance_ =
    nullptr;

const TestDependencyFactory& TestDependencyFactory::GetInstance() {
  RTC_DCHECK(IsValidTestDependencyFactoryThread());
  if (instance_ == nullptr) {
    instance_ = std::make_unique<TestDependencyFactory>();
  }
  return *instance_;
}

void TestDependencyFactory::SetInstance(
    std::unique_ptr<TestDependencyFactory> instance) {
  RTC_DCHECK(IsValidTestDependencyFactoryThread());
  RTC_CHECK(instance_ == nullptr);
  instance_ = std::move(instance);
}

std::unique_ptr<VideoQualityTestFixtureInterface::InjectionComponents>
TestDependencyFactory::CreateComponents() const {
  RTC_DCHECK(IsValidTestDependencyFactoryThread());
  return nullptr;
}

}  // namespace webrtc
