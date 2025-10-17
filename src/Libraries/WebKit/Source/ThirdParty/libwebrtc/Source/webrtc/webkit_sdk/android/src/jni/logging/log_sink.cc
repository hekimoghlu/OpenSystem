/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 9, 2022.
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
#include "sdk/android/src/jni/logging/log_sink.h"

#include "absl/strings/string_view.h"
#include "sdk/android/generated_logging_jni/JNILogging_jni.h"

namespace webrtc {
namespace jni {

JNILogSink::JNILogSink(JNIEnv* env, const JavaRef<jobject>& j_logging)
    : j_logging_(env, j_logging) {}
JNILogSink::~JNILogSink() = default;

void JNILogSink::OnLogMessage(const std::string& msg) {
  RTC_DCHECK_NOTREACHED();
}

void JNILogSink::OnLogMessage(const std::string& msg,
                              rtc::LoggingSeverity severity,
                              const char* tag) {
  OnLogMessage(absl::string_view{msg}, severity, tag);
}

void JNILogSink::OnLogMessage(absl::string_view msg,
                              rtc::LoggingSeverity severity,
                              const char* tag) {
  JNIEnv* env = AttachCurrentThreadIfNeeded();
  Java_JNILogging_logToInjectable(
      env, j_logging_, NativeToJavaString(env, std::string(msg)),
      NativeToJavaInteger(env, severity), NativeToJavaString(env, tag));
}

}  // namespace jni
}  // namespace webrtc
