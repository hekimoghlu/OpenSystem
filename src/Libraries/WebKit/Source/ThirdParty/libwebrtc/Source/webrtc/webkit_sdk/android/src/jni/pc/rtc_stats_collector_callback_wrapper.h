/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 24, 2022.
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
#ifndef SDK_ANDROID_SRC_JNI_PC_RTC_STATS_COLLECTOR_CALLBACK_WRAPPER_H_
#define SDK_ANDROID_SRC_JNI_PC_RTC_STATS_COLLECTOR_CALLBACK_WRAPPER_H_

#include <jni.h>

#include "api/peer_connection_interface.h"
#include "sdk/android/src/jni/jni_helpers.h"

namespace webrtc {
namespace jni {

// Adapter for a Java RTCStatsCollectorCallback presenting a C++
// RTCStatsCollectorCallback and dispatching the callback from C++ back to
// Java.
class RTCStatsCollectorCallbackWrapper : public RTCStatsCollectorCallback {
 public:
  RTCStatsCollectorCallbackWrapper(JNIEnv* jni,
                                   const JavaRef<jobject>& j_callback);
  ~RTCStatsCollectorCallbackWrapper() override;

  void OnStatsDelivered(
      const rtc::scoped_refptr<const RTCStatsReport>& report) override;

 private:
  const ScopedJavaGlobalRef<jobject> j_callback_global_;
};

}  // namespace jni
}  // namespace webrtc

#endif  // SDK_ANDROID_SRC_JNI_PC_RTC_STATS_COLLECTOR_CALLBACK_WRAPPER_H_
