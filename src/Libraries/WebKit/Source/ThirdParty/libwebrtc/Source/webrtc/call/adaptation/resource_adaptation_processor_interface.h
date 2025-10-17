/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 12, 2024.
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
#ifndef CALL_ADAPTATION_RESOURCE_ADAPTATION_PROCESSOR_INTERFACE_H_
#define CALL_ADAPTATION_RESOURCE_ADAPTATION_PROCESSOR_INTERFACE_H_

#include <map>
#include <optional>
#include <vector>

#include "api/adaptation/resource.h"
#include "api/rtp_parameters.h"
#include "api/scoped_refptr.h"
#include "api/task_queue/task_queue_base.h"
#include "api/video/video_adaptation_counters.h"
#include "api/video/video_frame.h"
#include "call/adaptation/adaptation_constraint.h"
#include "call/adaptation/encoder_settings.h"
#include "call/adaptation/video_source_restrictions.h"

namespace webrtc {

class ResourceLimitationsListener {
 public:
  virtual ~ResourceLimitationsListener();

  // The limitations on a resource were changed. This does not mean the current
  // video restrictions have changed.
  virtual void OnResourceLimitationChanged(
      rtc::scoped_refptr<Resource> resource,
      const std::map<rtc::scoped_refptr<Resource>, VideoAdaptationCounters>&
          resource_limitations) = 0;
};

// The Resource Adaptation Processor is responsible for reacting to resource
// usage measurements (e.g. overusing or underusing CPU). When a resource is
// overused the Processor is responsible for performing mitigations in order to
// consume less resources.
class ResourceAdaptationProcessorInterface {
 public:
  virtual ~ResourceAdaptationProcessorInterface();

  virtual void AddResourceLimitationsListener(
      ResourceLimitationsListener* limitations_listener) = 0;
  virtual void RemoveResourceLimitationsListener(
      ResourceLimitationsListener* limitations_listener) = 0;
  // Starts or stops listening to resources, effectively enabling or disabling
  // processing. May be called from anywhere.
  // TODO(https://crbug.com/webrtc/11172): Automatically register and unregister
  // with AddResource() and RemoveResource() instead. When the processor is
  // multi-stream aware, stream-specific resouces will get added and removed
  // over time.
  virtual void AddResource(rtc::scoped_refptr<Resource> resource) = 0;
  virtual std::vector<rtc::scoped_refptr<Resource>> GetResources() const = 0;
  virtual void RemoveResource(rtc::scoped_refptr<Resource> resource) = 0;
};

}  // namespace webrtc

#endif  // CALL_ADAPTATION_RESOURCE_ADAPTATION_PROCESSOR_INTERFACE_H_
