/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 20, 2023.
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
#ifndef RTC_BASE_EVENT_TRACER_H_
#define RTC_BASE_EVENT_TRACER_H_

// This file defines the interface for event tracing in WebRTC.
//
// Event log handlers are set through SetupEventTracer(). User of this API will
// provide two function pointers to handle event tracing calls.
//
// * GetCategoryEnabledPtr
//   Event tracing system calls this function to determine if a particular
//   event category is enabled.
//
// * AddTraceEventPtr
//   Adds a tracing event. It is the user's responsibility to log the data
//   provided.
//
// Parameters for the above two functions are described in trace_event.h.

#include <stdio.h>

#include "absl/strings/string_view.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

#if defined(RTC_USE_PERFETTO)
void RegisterPerfettoTrackEvents();
#else
typedef const unsigned char* (*GetCategoryEnabledPtr)(const char* name);
typedef void (*AddTraceEventPtr)(char phase,
                                 const unsigned char* category_enabled,
                                 const char* name,
                                 unsigned long long id,
                                 int num_args,
                                 const char** arg_names,
                                 const unsigned char* arg_types,
                                 const unsigned long long* arg_values,
                                 unsigned char flags);

// User of WebRTC can call this method to setup event tracing.
//
// This method must be called before any WebRTC methods. Functions
// provided should be thread-safe.
void SetupEventTracer(GetCategoryEnabledPtr get_category_enabled_ptr,
                      AddTraceEventPtr add_trace_event_ptr);

// This class defines interface for the event tracing system to call
// internally. Do not call these methods directly.
class EventTracer {
 public:
  static const unsigned char* GetCategoryEnabled(const char* name);

  static void AddTraceEvent(char phase,
                            const unsigned char* category_enabled,
                            const char* name,
                            unsigned long long id,
                            int num_args,
                            const char** arg_names,
                            const unsigned char* arg_types,
                            const unsigned long long* arg_values,
                            unsigned char flags);
};
#endif

}  // namespace webrtc

namespace rtc::tracing {
// Set up internal event tracer.
// TODO(webrtc:15917): Implement for perfetto.
RTC_EXPORT void SetupInternalTracer(bool enable_all_categories = true);
RTC_EXPORT bool StartInternalCapture(absl::string_view filename);
RTC_EXPORT void StartInternalCaptureToFile(FILE* file);
RTC_EXPORT void StopInternalCapture();
// Make sure we run this, this will tear down the internal tracing.
RTC_EXPORT void ShutdownInternalTracer();
}  // namespace rtc::tracing

#endif  // RTC_BASE_EVENT_TRACER_H_
