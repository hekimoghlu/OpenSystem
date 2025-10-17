/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 1, 2023.
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
#include "rtc_base/logging.h"

#include <memory>

#include "sdk/android/native_api/jni/java_types.h"
#include "sdk/android/src/jni/jni_helpers.h"

namespace webrtc {
namespace jni {

JNI_FUNCTION_DECLARATION(void,
                         Logging_nativeEnableLogToDebugOutput,
                         JNIEnv* jni,
                         jclass,
                         jint nativeSeverity) {
  if (nativeSeverity >= rtc::LS_VERBOSE && nativeSeverity <= rtc::LS_NONE) {
    rtc::LogMessage::LogToDebug(
        static_cast<rtc::LoggingSeverity>(nativeSeverity));
  }
}

JNI_FUNCTION_DECLARATION(void,
                         Logging_nativeEnableLogThreads,
                         JNIEnv* jni,
                         jclass) {
  rtc::LogMessage::LogThreads(true);
}

JNI_FUNCTION_DECLARATION(void,
                         Logging_nativeEnableLogTimeStamps,
                         JNIEnv* jni,
                         jclass) {
  rtc::LogMessage::LogTimestamps(true);
}

JNI_FUNCTION_DECLARATION(void,
                         Logging_nativeLog,
                         JNIEnv* jni,
                         jclass,
                         jint j_severity,
                         jstring j_tag,
                         jstring j_message) {
  std::string message = JavaToStdString(jni, JavaParamRef<jstring>(j_message));
  std::string tag = JavaToStdString(jni, JavaParamRef<jstring>(j_tag));
  RTC_LOG_TAG(static_cast<rtc::LoggingSeverity>(j_severity), tag.c_str())
      << message;
}

}  // namespace jni
}  // namespace webrtc
