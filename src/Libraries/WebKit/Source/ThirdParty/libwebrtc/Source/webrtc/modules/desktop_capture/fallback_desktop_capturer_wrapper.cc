/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 12, 2022.
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
#include "modules/desktop_capture/fallback_desktop_capturer_wrapper.h"

#include <stddef.h>

#include <utility>

#include "api/sequence_checker.h"
#include "rtc_base/checks.h"
#include "system_wrappers/include/metrics.h"

namespace webrtc {

namespace {

// Implementation to share a SharedMemoryFactory between DesktopCapturer
// instances. This class is designed for synchronized DesktopCapturer
// implementations only.
class SharedMemoryFactoryProxy : public SharedMemoryFactory {
 public:
  // Users should maintain the lifetime of `factory` to ensure it overlives
  // current instance.
  static std::unique_ptr<SharedMemoryFactory> Create(
      SharedMemoryFactory* factory);
  ~SharedMemoryFactoryProxy() override;

  // Forwards CreateSharedMemory() calls to `factory_`. Users should always call
  // this function in one thread. Users should not call this function after the
  // SharedMemoryFactory which current instance created from has been destroyed.
  std::unique_ptr<SharedMemory> CreateSharedMemory(size_t size) override;

 private:
  explicit SharedMemoryFactoryProxy(SharedMemoryFactory* factory);

  SharedMemoryFactory* factory_ = nullptr;
  SequenceChecker thread_checker_;
};

}  // namespace

SharedMemoryFactoryProxy::SharedMemoryFactoryProxy(
    SharedMemoryFactory* factory) {
  RTC_DCHECK(factory);
  factory_ = factory;
}

// static
std::unique_ptr<SharedMemoryFactory> SharedMemoryFactoryProxy::Create(
    SharedMemoryFactory* factory) {
  return std::unique_ptr<SharedMemoryFactory>(
      new SharedMemoryFactoryProxy(factory));
}

SharedMemoryFactoryProxy::~SharedMemoryFactoryProxy() = default;

std::unique_ptr<SharedMemory> SharedMemoryFactoryProxy::CreateSharedMemory(
    size_t size) {
  RTC_DCHECK(thread_checker_.IsCurrent());
  return factory_->CreateSharedMemory(size);
}

FallbackDesktopCapturerWrapper::FallbackDesktopCapturerWrapper(
    std::unique_ptr<DesktopCapturer> main_capturer,
    std::unique_ptr<DesktopCapturer> secondary_capturer)
    : main_capturer_(std::move(main_capturer)),
      secondary_capturer_(std::move(secondary_capturer)) {
  RTC_DCHECK(main_capturer_);
  RTC_DCHECK(secondary_capturer_);
}

FallbackDesktopCapturerWrapper::~FallbackDesktopCapturerWrapper() = default;

void FallbackDesktopCapturerWrapper::Start(
    DesktopCapturer::Callback* callback) {
  callback_ = callback;
  // FallbackDesktopCapturerWrapper catchs the callback of the main capturer,
  // and checks its return value to decide whether the secondary capturer should
  // be involved.
  main_capturer_->Start(this);
  // For the secondary capturer, we do not have a backup plan anymore, so
  // FallbackDesktopCapturerWrapper won't check its return value any more. It
  // will directly return to the input `callback`.
  secondary_capturer_->Start(callback);
}

void FallbackDesktopCapturerWrapper::SetSharedMemoryFactory(
    std::unique_ptr<SharedMemoryFactory> shared_memory_factory) {
  shared_memory_factory_ = std::move(shared_memory_factory);
  if (shared_memory_factory_) {
    main_capturer_->SetSharedMemoryFactory(
        SharedMemoryFactoryProxy::Create(shared_memory_factory_.get()));
    secondary_capturer_->SetSharedMemoryFactory(
        SharedMemoryFactoryProxy::Create(shared_memory_factory_.get()));
  } else {
    main_capturer_->SetSharedMemoryFactory(
        std::unique_ptr<SharedMemoryFactory>());
    secondary_capturer_->SetSharedMemoryFactory(
        std::unique_ptr<SharedMemoryFactory>());
  }
}

void FallbackDesktopCapturerWrapper::CaptureFrame() {
  RTC_DCHECK(callback_);
  if (main_capturer_permanent_error_) {
    secondary_capturer_->CaptureFrame();
  } else {
    main_capturer_->CaptureFrame();
  }
}

void FallbackDesktopCapturerWrapper::SetExcludedWindow(WindowId window) {
  main_capturer_->SetExcludedWindow(window);
  secondary_capturer_->SetExcludedWindow(window);
}

bool FallbackDesktopCapturerWrapper::GetSourceList(SourceList* sources) {
  if (main_capturer_permanent_error_) {
    return secondary_capturer_->GetSourceList(sources);
  }
  return main_capturer_->GetSourceList(sources);
}

bool FallbackDesktopCapturerWrapper::SelectSource(SourceId id) {
  if (main_capturer_permanent_error_) {
    return secondary_capturer_->SelectSource(id);
  }
  const bool main_capturer_result = main_capturer_->SelectSource(id);
  RTC_HISTOGRAM_BOOLEAN(
      "WebRTC.DesktopCapture.PrimaryCapturerSelectSourceError",
      main_capturer_result);
  if (!main_capturer_result) {
    main_capturer_permanent_error_ = true;
  }

  return secondary_capturer_->SelectSource(id);
}

bool FallbackDesktopCapturerWrapper::FocusOnSelectedSource() {
  if (main_capturer_permanent_error_) {
    return secondary_capturer_->FocusOnSelectedSource();
  }
  return main_capturer_->FocusOnSelectedSource() ||
         secondary_capturer_->FocusOnSelectedSource();
}

bool FallbackDesktopCapturerWrapper::IsOccluded(const DesktopVector& pos) {
  // Returns true if either capturer returns true.
  if (main_capturer_permanent_error_) {
    return secondary_capturer_->IsOccluded(pos);
  }
  return main_capturer_->IsOccluded(pos) ||
         secondary_capturer_->IsOccluded(pos);
}

void FallbackDesktopCapturerWrapper::OnCaptureResult(
    Result result,
    std::unique_ptr<DesktopFrame> frame) {
  RTC_DCHECK(callback_);
  RTC_HISTOGRAM_BOOLEAN("WebRTC.DesktopCapture.PrimaryCapturerError",
                        result != Result::SUCCESS);
  RTC_HISTOGRAM_BOOLEAN("WebRTC.DesktopCapture.PrimaryCapturerPermanentError",
                        result == Result::ERROR_PERMANENT);
  if (result == Result::SUCCESS) {
    callback_->OnCaptureResult(result, std::move(frame));
    return;
  }

  if (result == Result::ERROR_PERMANENT) {
    main_capturer_permanent_error_ = true;
  }
  secondary_capturer_->CaptureFrame();
}

}  // namespace webrtc
