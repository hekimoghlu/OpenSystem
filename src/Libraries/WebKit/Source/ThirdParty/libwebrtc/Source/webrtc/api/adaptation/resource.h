/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 29, 2022.
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
#ifndef API_ADAPTATION_RESOURCE_H_
#define API_ADAPTATION_RESOURCE_H_

#include <string>

#include "api/ref_count.h"
#include "api/scoped_refptr.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

class Resource;

enum class ResourceUsageState {
  // Action is needed to minimze the load on this resource.
  kOveruse,
  // Increasing the load on this resource is desired, if possible.
  kUnderuse,
};

RTC_EXPORT const char* ResourceUsageStateToString(
    ResourceUsageState usage_state);

class RTC_EXPORT ResourceListener {
 public:
  virtual ~ResourceListener();

  virtual void OnResourceUsageStateMeasured(
      rtc::scoped_refptr<Resource> resource,
      ResourceUsageState usage_state) = 0;
};

// A Resource monitors an implementation-specific resource. It may report
// kOveruse or kUnderuse when resource usage is high or low enough that we
// should perform some sort of mitigation to fulfil the resource's constraints.
//
// The methods on this interface are invoked on the adaptation task queue.
// Resource usage measurements may be performed on an any task queue.
//
// The Resource is reference counted to prevent use-after-free when posting
// between task queues. As such, the implementation MUST NOT make any
// assumptions about which task queue Resource is destructed on.
class RTC_EXPORT Resource : public RefCountInterface {
 public:
  Resource();
  // Destruction may happen on any task queue.
  ~Resource() override;

  virtual std::string Name() const = 0;
  // The `listener` may be informed of resource usage measurements on any task
  // queue, but not after this method is invoked with the null argument.
  virtual void SetResourceListener(ResourceListener* listener) = 0;
};

}  // namespace webrtc

#endif  // API_ADAPTATION_RESOURCE_H_
