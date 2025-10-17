/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 25, 2025.
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
#include "modules/desktop_capture/win/dxgi_adapter_duplicator.h"

#include <comdef.h>
#include <dxgi.h>

#include <algorithm>

#include "modules/desktop_capture/win/desktop_capture_utils.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"

namespace webrtc {

using Microsoft::WRL::ComPtr;

namespace {

bool IsValidRect(const RECT& rect) {
  return rect.right > rect.left && rect.bottom > rect.top;
}

}  // namespace

DxgiAdapterDuplicator::DxgiAdapterDuplicator(const D3dDevice& device)
    : device_(device) {}
DxgiAdapterDuplicator::DxgiAdapterDuplicator(DxgiAdapterDuplicator&&) = default;
DxgiAdapterDuplicator::~DxgiAdapterDuplicator() = default;

bool DxgiAdapterDuplicator::Initialize() {
  if (DoInitialize()) {
    return true;
  }
  duplicators_.clear();
  return false;
}

bool DxgiAdapterDuplicator::DoInitialize() {
  for (int i = 0;; i++) {
    ComPtr<IDXGIOutput> output;
    _com_error error =
        device_.dxgi_adapter()->EnumOutputs(i, output.GetAddressOf());
    if (error.Error() == DXGI_ERROR_NOT_FOUND) {
      break;
    }

    if (error.Error() == DXGI_ERROR_NOT_CURRENTLY_AVAILABLE) {
      RTC_LOG(LS_WARNING) << "IDXGIAdapter::EnumOutputs returned "
                          << "NOT_CURRENTLY_AVAILABLE. This may happen when "
                          << "running in session 0.";
      break;
    }

    if (error.Error() != S_OK || !output) {
      RTC_LOG(LS_WARNING) << "IDXGIAdapter::EnumOutputs returned an unexpected "
                          << "result: "
                          << desktop_capture::utils::ComErrorToString(error);
      continue;
    }

    DXGI_OUTPUT_DESC desc;
    error = output->GetDesc(&desc);
    if (error.Error() == S_OK) {
      if (desc.AttachedToDesktop && IsValidRect(desc.DesktopCoordinates)) {
        ComPtr<IDXGIOutput1> output1;
        error = output.As(&output1);
        if (error.Error() != S_OK || !output1) {
          RTC_LOG(LS_WARNING)
              << "Failed to convert IDXGIOutput to IDXGIOutput1, this usually "
              << "means the system does not support DirectX 11";
          continue;
        }
        DxgiOutputDuplicator duplicator(device_, output1, desc);
        if (!duplicator.Initialize()) {
          RTC_LOG(LS_WARNING) << "Failed to initialize DxgiOutputDuplicator on "
                              << "output " << i;
          continue;
        }

        duplicators_.push_back(std::move(duplicator));
        desktop_rect_.UnionWith(duplicators_.back().desktop_rect());
      } else {
        RTC_LOG(LS_ERROR) << (desc.AttachedToDesktop ? "Attached" : "Detached")
                          << " output " << i << " ("
                          << desc.DesktopCoordinates.top << ", "
                          << desc.DesktopCoordinates.left << ") - ("
                          << desc.DesktopCoordinates.bottom << ", "
                          << desc.DesktopCoordinates.right << ") is ignored.";
      }
    } else {
      RTC_LOG(LS_WARNING) << "Failed to get output description of device " << i
                          << ", ignore.";
    }
  }

  if (duplicators_.empty()) {
    RTC_LOG(LS_WARNING)
        << "Cannot initialize any DxgiOutputDuplicator instance.";
  }

  return !duplicators_.empty();
}

void DxgiAdapterDuplicator::Setup(Context* context) {
  RTC_DCHECK(context->contexts.empty());
  context->contexts.resize(duplicators_.size());
  for (size_t i = 0; i < duplicators_.size(); i++) {
    duplicators_[i].Setup(&context->contexts[i]);
  }
}

void DxgiAdapterDuplicator::Unregister(const Context* const context) {
  RTC_DCHECK_EQ(context->contexts.size(), duplicators_.size());
  for (size_t i = 0; i < duplicators_.size(); i++) {
    duplicators_[i].Unregister(&context->contexts[i]);
  }
}

bool DxgiAdapterDuplicator::Duplicate(Context* context,
                                      SharedDesktopFrame* target) {
  RTC_DCHECK_EQ(context->contexts.size(), duplicators_.size());
  for (size_t i = 0; i < duplicators_.size(); i++) {
    if (!duplicators_[i].Duplicate(&context->contexts[i],
                                   duplicators_[i].desktop_rect().top_left(),
                                   target)) {
      return false;
    }
  }
  return true;
}

bool DxgiAdapterDuplicator::DuplicateMonitor(Context* context,
                                             int monitor_id,
                                             SharedDesktopFrame* target) {
  RTC_DCHECK_GE(monitor_id, 0);
  RTC_DCHECK_LT(monitor_id, duplicators_.size());
  RTC_DCHECK_EQ(context->contexts.size(), duplicators_.size());
  return duplicators_[monitor_id].Duplicate(&context->contexts[monitor_id],
                                            DesktopVector(), target);
}

DesktopRect DxgiAdapterDuplicator::ScreenRect(int id) const {
  RTC_DCHECK_GE(id, 0);
  RTC_DCHECK_LT(id, duplicators_.size());
  return duplicators_[id].desktop_rect();
}

const std::string& DxgiAdapterDuplicator::GetDeviceName(int id) const {
  RTC_DCHECK_GE(id, 0);
  RTC_DCHECK_LT(id, duplicators_.size());
  return duplicators_[id].device_name();
}

int DxgiAdapterDuplicator::screen_count() const {
  return static_cast<int>(duplicators_.size());
}

int64_t DxgiAdapterDuplicator::GetNumFramesCaptured() const {
  int64_t min = INT64_MAX;
  for (const auto& duplicator : duplicators_) {
    min = std::min(min, duplicator.num_frames_captured());
  }

  return min;
}

void DxgiAdapterDuplicator::TranslateRect(const DesktopVector& position) {
  desktop_rect_.Translate(position);
  RTC_DCHECK_GE(desktop_rect_.left(), 0);
  RTC_DCHECK_GE(desktop_rect_.top(), 0);
  for (auto& duplicator : duplicators_) {
    duplicator.TranslateRect(position);
  }
}

}  // namespace webrtc
