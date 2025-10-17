/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 22, 2025.
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
#include <map>
#include <memory>

#include "rtc_base/string_utils.h"
#include "sdk/android/generated_metrics_jni/Metrics_jni.h"
#include "sdk/android/native_api/jni/java_types.h"
#include "sdk/android/src/jni/jni_helpers.h"
#include "system_wrappers/include/metrics.h"

// Enables collection of native histograms and creating them.
namespace webrtc {
namespace jni {

static void JNI_Metrics_Enable(JNIEnv* jni) {
  metrics::Enable();
}

// Gets and clears native histograms.
static ScopedJavaLocalRef<jobject> JNI_Metrics_GetAndReset(JNIEnv* jni) {
  ScopedJavaLocalRef<jobject> j_metrics = Java_Metrics_Constructor(jni);

  std::map<std::string, std::unique_ptr<metrics::SampleInfo>,
           rtc::AbslStringViewCmp>
      histograms;
  metrics::GetAndReset(&histograms);
  for (const auto& kv : histograms) {
    // Create and add samples to `HistogramInfo`.
    ScopedJavaLocalRef<jobject> j_info = Java_HistogramInfo_Constructor(
        jni, kv.second->min, kv.second->max,
        static_cast<int>(kv.second->bucket_count));
    for (const auto& sample : kv.second->samples) {
      Java_HistogramInfo_addSample(jni, j_info, sample.first, sample.second);
    }
    // Add `HistogramInfo` to `Metrics`.
    ScopedJavaLocalRef<jstring> j_name = NativeToJavaString(jni, kv.first);
    Java_Metrics_add(jni, j_metrics, j_name, j_info);
  }
  CHECK_EXCEPTION(jni);
  return j_metrics;
}

}  // namespace jni
}  // namespace webrtc
