/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 6, 2023.
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
#include "call/adaptation/broadcast_resource_listener.h"

#include <algorithm>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "api/make_ref_counted.h"
#include "rtc_base/checks.h"
#include "rtc_base/synchronization/mutex.h"

namespace webrtc {

// The AdapterResource redirects resource usage measurements from its parent to
// a single ResourceListener.
class BroadcastResourceListener::AdapterResource : public Resource {
 public:
  explicit AdapterResource(absl::string_view name) : name_(std::move(name)) {}
  ~AdapterResource() override { RTC_DCHECK(!listener_); }

  // The parent is letting us know we have a usage neasurement.
  void OnResourceUsageStateMeasured(ResourceUsageState usage_state) {
    MutexLock lock(&lock_);
    if (!listener_)
      return;
    listener_->OnResourceUsageStateMeasured(rtc::scoped_refptr<Resource>(this),
                                            usage_state);
  }

  // Resource implementation.
  std::string Name() const override { return name_; }
  void SetResourceListener(ResourceListener* listener) override {
    MutexLock lock(&lock_);
    RTC_DCHECK(!listener_ || !listener);
    listener_ = listener;
  }

 private:
  const std::string name_;
  Mutex lock_;
  ResourceListener* listener_ RTC_GUARDED_BY(lock_) = nullptr;
};

BroadcastResourceListener::BroadcastResourceListener(
    rtc::scoped_refptr<Resource> source_resource)
    : source_resource_(source_resource), is_listening_(false) {
  RTC_DCHECK(source_resource_);
}

BroadcastResourceListener::~BroadcastResourceListener() {
  RTC_DCHECK(!is_listening_);
}

rtc::scoped_refptr<Resource> BroadcastResourceListener::SourceResource() const {
  return source_resource_;
}

void BroadcastResourceListener::StartListening() {
  MutexLock lock(&lock_);
  RTC_DCHECK(!is_listening_);
  source_resource_->SetResourceListener(this);
  is_listening_ = true;
}

void BroadcastResourceListener::StopListening() {
  MutexLock lock(&lock_);
  RTC_DCHECK(is_listening_);
  RTC_DCHECK(adapters_.empty());
  source_resource_->SetResourceListener(nullptr);
  is_listening_ = false;
}

rtc::scoped_refptr<Resource>
BroadcastResourceListener::CreateAdapterResource() {
  MutexLock lock(&lock_);
  RTC_DCHECK(is_listening_);
  rtc::scoped_refptr<AdapterResource> adapter =
      rtc::make_ref_counted<AdapterResource>(source_resource_->Name() +
                                             "Adapter");
  adapters_.push_back(adapter);
  return adapter;
}

void BroadcastResourceListener::RemoveAdapterResource(
    rtc::scoped_refptr<Resource> resource) {
  MutexLock lock(&lock_);
  auto it = std::find(adapters_.begin(), adapters_.end(), resource);
  RTC_DCHECK(it != adapters_.end());
  adapters_.erase(it);
}

std::vector<rtc::scoped_refptr<Resource>>
BroadcastResourceListener::GetAdapterResources() {
  std::vector<rtc::scoped_refptr<Resource>> resources;
  MutexLock lock(&lock_);
  for (const auto& adapter : adapters_) {
    resources.push_back(adapter);
  }
  return resources;
}

void BroadcastResourceListener::OnResourceUsageStateMeasured(
    rtc::scoped_refptr<Resource> resource,
    ResourceUsageState usage_state) {
  RTC_DCHECK_EQ(resource, source_resource_);
  MutexLock lock(&lock_);
  for (const auto& adapter : adapters_) {
    adapter->OnResourceUsageStateMeasured(usage_state);
  }
}

}  // namespace webrtc
