/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 14, 2021.
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
#include "sdk/android/src/jni/pc/stats_observer.h"

#include <vector>

#include "sdk/android/generated_peerconnection_jni/StatsObserver_jni.h"
#include "sdk/android/generated_peerconnection_jni/StatsReport_jni.h"
#include "sdk/android/native_api/jni/java_types.h"
#include "sdk/android/src/jni/jni_helpers.h"

namespace webrtc {
namespace jni {

namespace {

ScopedJavaLocalRef<jobject> NativeToJavaStatsReportValue(
    JNIEnv* env,
    const rtc::scoped_refptr<StatsReport::Value>& value_ptr) {
  // Should we use the '.name' enum value here instead of converting the
  // name to a string?
  return Java_Value_Constructor(
      env, NativeToJavaString(env, value_ptr->display_name()),
      NativeToJavaString(env, value_ptr->ToString()));
}

ScopedJavaLocalRef<jobjectArray> NativeToJavaStatsReportValueArray(
    JNIEnv* env,
    const StatsReport::Values& value_map) {
  // Ignore the keys and make an array out of the values.
  std::vector<StatsReport::ValuePtr> values;
  for (const auto& it : value_map)
    values.push_back(it.second);
  return NativeToJavaObjectArray(env, values,
                                 org_webrtc_StatsReport_00024Value_clazz(env),
                                 &NativeToJavaStatsReportValue);
}

ScopedJavaLocalRef<jobject> NativeToJavaStatsReport(JNIEnv* env,
                                                    const StatsReport& report) {
  return Java_StatsReport_Constructor(
      env, NativeToJavaString(env, report.id()->ToString()),
      NativeToJavaString(env, report.TypeToString()), report.timestamp(),
      NativeToJavaStatsReportValueArray(env, report.values()));
}

}  // namespace

StatsObserverJni::StatsObserverJni(JNIEnv* jni,
                                   const JavaRef<jobject>& j_observer)
    : j_observer_global_(jni, j_observer) {}

StatsObserverJni::~StatsObserverJni() = default;

void StatsObserverJni::OnComplete(const StatsReports& reports) {
  JNIEnv* env = AttachCurrentThreadIfNeeded();
  ScopedJavaLocalRef<jobjectArray> j_reports =
      NativeToJavaObjectArray(env, reports, org_webrtc_StatsReport_clazz(env),
                              [](JNIEnv* env, const StatsReport* report) {
                                return NativeToJavaStatsReport(env, *report);
                              });
  Java_StatsObserver_onComplete(env, j_observer_global_, j_reports);
}

}  // namespace jni
}  // namespace webrtc
