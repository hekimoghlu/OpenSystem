/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 7, 2025.
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
#ifndef CALL_ADAPTATION_BROADCAST_RESOURCE_LISTENER_H_
#define CALL_ADAPTATION_BROADCAST_RESOURCE_LISTENER_H_

#include <vector>

#include "api/adaptation/resource.h"
#include "api/scoped_refptr.h"
#include "rtc_base/synchronization/mutex.h"

namespace webrtc {

// Responsible for forwarding 1 resource usage measurement to N listeners by
// creating N "adapter" resources.
//
// Example:
// If we have ResourceA, ResourceListenerX and ResourceListenerY we can create a
// BroadcastResourceListener that listens to ResourceA, use CreateAdapter() to
// spawn adapter resources ResourceX and ResourceY and let ResourceListenerX
// listen to ResourceX and ResourceListenerY listen to ResourceY. When ResourceA
// makes a measurement it will be echoed by both ResourceX and ResourceY.
//
// TODO(https://crbug.com/webrtc/11565): When the ResourceAdaptationProcessor is
// moved to call there will only be one ResourceAdaptationProcessor that needs
// to listen to the injected resources. When this is the case, delete this class
// and DCHECK that a Resource's listener is never overwritten.
class BroadcastResourceListener : public ResourceListener {
 public:
  explicit BroadcastResourceListener(
      rtc::scoped_refptr<Resource> source_resource);
  ~BroadcastResourceListener() override;

  rtc::scoped_refptr<Resource> SourceResource() const;
  void StartListening();
  void StopListening();

  // Creates a Resource that redirects any resource usage measurements that
  // BroadcastResourceListener receives to its listener.
  rtc::scoped_refptr<Resource> CreateAdapterResource();

  // Unregister the adapter from the BroadcastResourceListener; it will no
  // longer receive resource usage measurement and will no longer be referenced.
  // Use this to prevent memory leaks of old adapters.
  void RemoveAdapterResource(rtc::scoped_refptr<Resource> resource);
  std::vector<rtc::scoped_refptr<Resource>> GetAdapterResources();

  // ResourceListener implementation.
  void OnResourceUsageStateMeasured(rtc::scoped_refptr<Resource> resource,
                                    ResourceUsageState usage_state) override;

 private:
  class AdapterResource;
  friend class AdapterResource;

  const rtc::scoped_refptr<Resource> source_resource_;
  Mutex lock_;
  bool is_listening_ RTC_GUARDED_BY(lock_);
  // The AdapterResource unregisters itself prior to destruction, guaranteeing
  // that these pointers are safe to use.
  std::vector<rtc::scoped_refptr<AdapterResource>> adapters_
      RTC_GUARDED_BY(lock_);
};

}  // namespace webrtc

#endif  // CALL_ADAPTATION_BROADCAST_RESOURCE_LISTENER_H_
