/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 16, 2025.
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
#include "rtc_base/log_sinks.h"
#include "sdk/android/generated_peerconnection_jni/CallSessionFileRotatingLogSink_jni.h"
#include "sdk/android/native_api/jni/java_types.h"
#include "sdk/android/src/jni/jni_helpers.h"

namespace webrtc {
namespace jni {

static jlong JNI_CallSessionFileRotatingLogSink_AddSink(
    JNIEnv* jni,
    const JavaParamRef<jstring>& j_dirPath,
    jint j_maxFileSize,
    jint j_severity) {
  std::string dir_path = JavaToStdString(jni, j_dirPath);
  rtc::CallSessionFileRotatingLogSink* sink =
      new rtc::CallSessionFileRotatingLogSink(dir_path, j_maxFileSize);
  if (!sink->Init()) {
    RTC_LOG_V(rtc::LoggingSeverity::LS_WARNING)
        << "Failed to init CallSessionFileRotatingLogSink for path "
        << dir_path;
    delete sink;
    return 0;
  }
  rtc::LogMessage::AddLogToStream(
      sink, static_cast<rtc::LoggingSeverity>(j_severity));
  return jlongFromPointer(sink);
}

static void JNI_CallSessionFileRotatingLogSink_DeleteSink(JNIEnv* jni,
                                                          jlong j_sink) {
  rtc::CallSessionFileRotatingLogSink* sink =
      reinterpret_cast<rtc::CallSessionFileRotatingLogSink*>(j_sink);
  rtc::LogMessage::RemoveLogToStream(sink);
  delete sink;
}

static ScopedJavaLocalRef<jbyteArray>
JNI_CallSessionFileRotatingLogSink_GetLogData(
    JNIEnv* jni,
    const JavaParamRef<jstring>& j_dirPath) {
  std::string dir_path = JavaToStdString(jni, j_dirPath);
  rtc::CallSessionFileRotatingStreamReader file_reader(dir_path);
  size_t log_size = file_reader.GetSize();
  if (log_size == 0) {
    RTC_LOG_V(rtc::LoggingSeverity::LS_WARNING)
        << "CallSessionFileRotatingStream returns 0 size for path " << dir_path;
    return ScopedJavaLocalRef<jbyteArray>(jni, jni->NewByteArray(0));
  }

  // TODO(nisse, sakal): To avoid copying, change api to use ByteBuffer.
  std::unique_ptr<jbyte> buffer(static_cast<jbyte*>(malloc(log_size)));
  size_t read = file_reader.ReadAll(buffer.get(), log_size);

  ScopedJavaLocalRef<jbyteArray> result =
      ScopedJavaLocalRef<jbyteArray>(jni, jni->NewByteArray(read));
  jni->SetByteArrayRegion(result.obj(), 0, read, buffer.get());

  return result;
}

}  // namespace jni
}  // namespace webrtc
