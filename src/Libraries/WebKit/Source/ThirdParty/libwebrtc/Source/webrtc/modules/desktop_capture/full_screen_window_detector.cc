/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 1, 2022.
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
#include "modules/desktop_capture/full_screen_window_detector.h"

#include "modules/desktop_capture/full_screen_application_handler.h"
#include "rtc_base/time_utils.h"

namespace webrtc {

FullScreenWindowDetector::FullScreenWindowDetector(
    ApplicationHandlerFactory application_handler_factory)
    : application_handler_factory_(application_handler_factory),
      last_update_time_ms_(0),
      previous_source_id_(0),
      no_handler_source_id_(0) {}

DesktopCapturer::SourceId FullScreenWindowDetector::FindFullScreenWindow(
    DesktopCapturer::SourceId original_source_id) {
  if (app_handler_ == nullptr ||
      app_handler_->GetSourceId() != original_source_id) {
    return 0;
  }
  return app_handler_->FindFullScreenWindow(window_list_, last_update_time_ms_);
}

void FullScreenWindowDetector::UpdateWindowListIfNeeded(
    DesktopCapturer::SourceId original_source_id,
    rtc::FunctionView<bool(DesktopCapturer::SourceList*)> get_sources) {
  const bool skip_update = previous_source_id_ != original_source_id;
  previous_source_id_ = original_source_id;

  // Here is an attempt to avoid redundant creating application handler in case
  // when an instance of WindowCapturer is used to generate a thumbnail to show
  // in picker by calling SelectSource and CaptureFrame for every available
  // source.
  if (skip_update) {
    return;
  }

  CreateApplicationHandlerIfNeeded(original_source_id);
  if (app_handler_ == nullptr) {
    // There is no FullScreenApplicationHandler specific for
    // current application
    return;
  }

  constexpr int64_t kUpdateIntervalMs = 500;

  if ((rtc::TimeMillis() - last_update_time_ms_) <= kUpdateIntervalMs) {
    return;
  }

  DesktopCapturer::SourceList window_list;
  if (get_sources(&window_list)) {
    last_update_time_ms_ = rtc::TimeMillis();
    window_list_.swap(window_list);
  }
}

void FullScreenWindowDetector::CreateApplicationHandlerIfNeeded(
    DesktopCapturer::SourceId source_id) {
  if (no_handler_source_id_ == source_id) {
    return;
  }

  if (app_handler_ == nullptr || app_handler_->GetSourceId() != source_id) {
    app_handler_ = application_handler_factory_
                       ? application_handler_factory_(source_id)
                       : nullptr;
  }

  if (app_handler_ == nullptr) {
    no_handler_source_id_ = source_id;
  }
}

}  // namespace webrtc
