/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 6, 2024.
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
#include "sdk/android/src/jni/pc/rtc_stats_collector_callback_wrapper.h"

#include <string>
#include <vector>

#include "rtc_base/string_encode.h"
#include "sdk/android/generated_external_classes_jni/BigInteger_jni.h"
#include "sdk/android/generated_peerconnection_jni/RTCStatsCollectorCallback_jni.h"
#include "sdk/android/generated_peerconnection_jni/RTCStatsReport_jni.h"
#include "sdk/android/generated_peerconnection_jni/RTCStats_jni.h"
#include "sdk/android/native_api/jni/java_types.h"

namespace webrtc {
namespace jni {

namespace {

ScopedJavaLocalRef<jobject> NativeToJavaBigInteger(JNIEnv* env, uint64_t u) {
#ifdef RTC_JNI_GENERATOR_LEGACY_SYMBOLS
  return JNI_BigInteger::Java_BigInteger_ConstructorJMBI_JLS(
      env, NativeToJavaString(env, rtc::ToString(u)));
#else
  return JNI_BigInteger::Java_BigInteger_Constructor__String(
      env, NativeToJavaString(env, rtc::ToString(u)));
#endif
}

ScopedJavaLocalRef<jobjectArray> NativeToJavaBigIntegerArray(
    JNIEnv* env,
    const std::vector<uint64_t>& container) {
  return NativeToJavaObjectArray(
      env, container, java_math_BigInteger_clazz(env), &NativeToJavaBigInteger);
}

ScopedJavaLocalRef<jobject> MemberToJava(
    JNIEnv* env,
    const RTCStatsMemberInterface& member) {
  switch (member.type()) {
    case RTCStatsMemberInterface::kBool:
      return NativeToJavaBoolean(env, *member.cast_to<RTCStatsMember<bool>>());

    case RTCStatsMemberInterface::kInt32:
      return NativeToJavaInteger(env,
                                 *member.cast_to<RTCStatsMember<int32_t>>());

    case RTCStatsMemberInterface::kUint32:
      return NativeToJavaLong(env, *member.cast_to<RTCStatsMember<uint32_t>>());

    case RTCStatsMemberInterface::kInt64:
      return NativeToJavaLong(env, *member.cast_to<RTCStatsMember<int64_t>>());

    case RTCStatsMemberInterface::kUint64:
      return NativeToJavaBigInteger(
          env, *member.cast_to<RTCStatsMember<uint64_t>>());

    case RTCStatsMemberInterface::kDouble:
      return NativeToJavaDouble(env, *member.cast_to<RTCStatsMember<double>>());

    case RTCStatsMemberInterface::kString:
      return NativeToJavaString(env,
                                *member.cast_to<RTCStatsMember<std::string>>());

    case RTCStatsMemberInterface::kSequenceBool:
      return NativeToJavaBooleanArray(
          env, *member.cast_to<RTCStatsMember<std::vector<bool>>>());

    case RTCStatsMemberInterface::kSequenceInt32:
      return NativeToJavaIntegerArray(
          env, *member.cast_to<RTCStatsMember<std::vector<int32_t>>>());

    case RTCStatsMemberInterface::kSequenceUint32: {
      const std::vector<uint32_t>& v =
          *member.cast_to<RTCStatsMember<std::vector<uint32_t>>>();
      return NativeToJavaLongArray(env,
                                   std::vector<int64_t>(v.begin(), v.end()));
    }
    case RTCStatsMemberInterface::kSequenceInt64:
      return NativeToJavaLongArray(
          env, *member.cast_to<RTCStatsMember<std::vector<int64_t>>>());

    case RTCStatsMemberInterface::kSequenceUint64:
      return NativeToJavaBigIntegerArray(
          env, *member.cast_to<RTCStatsMember<std::vector<uint64_t>>>());

    case RTCStatsMemberInterface::kSequenceDouble:
      return NativeToJavaDoubleArray(
          env, *member.cast_to<RTCStatsMember<std::vector<double>>>());

    case RTCStatsMemberInterface::kSequenceString:
      return NativeToJavaStringArray(
          env, *member.cast_to<RTCStatsMember<std::vector<std::string>>>());

    case RTCStatsMemberInterface::kMapStringUint64:
      return NativeToJavaMap(
          env,
          *member.cast_to<RTCStatsMember<std::map<std::string, uint64_t>>>(),
          [](JNIEnv* env, const auto& entry) {
            return std::make_pair(NativeToJavaString(env, entry.first),
                                  NativeToJavaBigInteger(env, entry.second));
          });

    case RTCStatsMemberInterface::kMapStringDouble:
      return NativeToJavaMap(
          env, *member.cast_to<RTCStatsMember<std::map<std::string, double>>>(),
          [](JNIEnv* env, const auto& entry) {
            return std::make_pair(NativeToJavaString(env, entry.first),
                                  NativeToJavaDouble(env, entry.second));
          });
  }
  RTC_DCHECK_NOTREACHED();
  return nullptr;
}

ScopedJavaLocalRef<jobject> NativeToJavaRtcStats(JNIEnv* env,
                                                 const RTCStats& stats) {
  JavaMapBuilder builder(env);
  for (auto* const member : stats.Members()) {
    if (!member->is_defined())
      continue;
    builder.put(NativeToJavaString(env, member->name()),
                MemberToJava(env, *member));
  }
  return Java_RTCStats_create(
      env, stats.timestamp().us(), NativeToJavaString(env, stats.type()),
      NativeToJavaString(env, stats.id()), builder.GetJavaMap());
}

ScopedJavaLocalRef<jobject> NativeToJavaRtcStatsReport(
    JNIEnv* env,
    const rtc::scoped_refptr<const RTCStatsReport>& report) {
  ScopedJavaLocalRef<jobject> j_stats_map =
      NativeToJavaMap(env, *report, [](JNIEnv* env, const RTCStats& stats) {
        return std::make_pair(NativeToJavaString(env, stats.id()),
                              NativeToJavaRtcStats(env, stats));
      });
  return Java_RTCStatsReport_create(env, report->timestamp().us(), j_stats_map);
}

}  // namespace

RTCStatsCollectorCallbackWrapper::RTCStatsCollectorCallbackWrapper(
    JNIEnv* jni,
    const JavaRef<jobject>& j_callback)
    : j_callback_global_(jni, j_callback) {}

RTCStatsCollectorCallbackWrapper::~RTCStatsCollectorCallbackWrapper() = default;

void RTCStatsCollectorCallbackWrapper::OnStatsDelivered(
    const rtc::scoped_refptr<const RTCStatsReport>& report) {
  JNIEnv* jni = AttachCurrentThreadIfNeeded();
  Java_RTCStatsCollectorCallback_onStatsDelivered(
      jni, j_callback_global_, NativeToJavaRtcStatsReport(jni, report));
}

}  // namespace jni
}  // namespace webrtc
