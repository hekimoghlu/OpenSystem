/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 30, 2024.
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
#include "rtc_base/timestamp_aligner.h"

#include <jni.h>

#include "rtc_base/time_utils.h"
#include "sdk/android/generated_video_jni/TimestampAligner_jni.h"
#include "sdk/android/src/jni/jni_helpers.h"

namespace webrtc {
namespace jni {

static jlong JNI_TimestampAligner_RtcTimeNanos(JNIEnv* env) {
  return rtc::TimeNanos();
}

static jlong JNI_TimestampAligner_CreateTimestampAligner(JNIEnv* env) {
  return jlongFromPointer(new rtc::TimestampAligner());
}

static void JNI_TimestampAligner_ReleaseTimestampAligner(
    JNIEnv* env,
    jlong timestamp_aligner) {
  delete reinterpret_cast<rtc::TimestampAligner*>(timestamp_aligner);
}

static jlong JNI_TimestampAligner_TranslateTimestamp(JNIEnv* env,
                                                     jlong timestamp_aligner,
                                                     jlong camera_time_ns) {
  return reinterpret_cast<rtc::TimestampAligner*>(timestamp_aligner)
             ->TranslateTimestamp(camera_time_ns / rtc::kNumNanosecsPerMicrosec,
                                  rtc::TimeMicros()) *
         rtc::kNumNanosecsPerMicrosec;
}

}  // namespace jni
}  // namespace webrtc
