/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 23, 2025.
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
#include "modules/video_coding/codecs/test/android_codec_factory_helper.h"

#include <jni.h>
#include <pthread.h>
#include <stddef.h>

#include <memory>

#include "modules/utility/include/jvm_android.h"
#include "rtc_base/checks.h"
#include "sdk/android/native_api/codecs/wrapper.h"
#include "sdk/android/native_api/jni/class_loader.h"
#include "sdk/android/native_api/jni/jvm.h"
#include "sdk/android/native_api/jni/scoped_java_ref.h"
#include "sdk/android/src/jni/jvm.h"

namespace webrtc {
namespace test {

namespace {

static pthread_once_t g_initialize_once = PTHREAD_ONCE_INIT;

void EnsureInitializedOnce() {
  RTC_CHECK(::webrtc::jni::GetJVM() != nullptr);

  JNIEnv* jni = ::webrtc::jni::AttachCurrentThreadIfNeeded();
  JavaVM* jvm = NULL;
  RTC_CHECK_EQ(0, jni->GetJavaVM(&jvm));

  // Initialize the Java environment (currently only used by the audio manager).
  webrtc::JVM::Initialize(jvm);
}

}  // namespace

void InitializeAndroidObjects() {
  RTC_CHECK_EQ(0, pthread_once(&g_initialize_once, &EnsureInitializedOnce));
}

std::unique_ptr<VideoEncoderFactory> CreateAndroidEncoderFactory() {
  JNIEnv* env = AttachCurrentThreadIfNeeded();
  ScopedJavaLocalRef<jclass> factory_class =
      GetClass(env, "org/webrtc/HardwareVideoEncoderFactory");
  jmethodID factory_constructor = env->GetMethodID(
      factory_class.obj(), "<init>", "(Lorg/webrtc/EglBase$Context;ZZ)V");
  ScopedJavaLocalRef<jobject> factory_object(
      env, env->NewObject(factory_class.obj(), factory_constructor,
                          nullptr /* shared_context */,
                          false /* enable_intel_vp8_encoder */,
                          true /* enable_h264_high_profile */));
  return JavaToNativeVideoEncoderFactory(env, factory_object.obj());
}

std::unique_ptr<VideoDecoderFactory> CreateAndroidDecoderFactory() {
  JNIEnv* env = AttachCurrentThreadIfNeeded();
  ScopedJavaLocalRef<jclass> factory_class =
      GetClass(env, "org/webrtc/HardwareVideoDecoderFactory");
  jmethodID factory_constructor = env->GetMethodID(
      factory_class.obj(), "<init>", "(Lorg/webrtc/EglBase$Context;)V");
  ScopedJavaLocalRef<jobject> factory_object(
      env, env->NewObject(factory_class.obj(), factory_constructor,
                          nullptr /* shared_context */));
  return JavaToNativeVideoDecoderFactory(env, factory_object.obj());
}

}  // namespace test
}  // namespace webrtc
