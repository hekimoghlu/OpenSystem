/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 4, 2025.
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
#include "pc/test/android_test_initializer.h"

#include <jni.h>
#include <pthread.h>
#include <stddef.h>

#include "modules/utility/include/jvm_android.h"
#include "rtc_base/checks.h"
#include "sdk/android/src/jni/jvm.h"
// TODO(phoglund): This include is to a target we can't really depend on.
// We need to either break it out into a smaller target or find some way to
// not use it.
#include "rtc_base/ssl_adapter.h"

namespace webrtc {

namespace {

static pthread_once_t g_initialize_once = PTHREAD_ONCE_INIT;

// There can only be one JNI_OnLoad in each binary. So since this is a GTEST
// C++ runner binary, we want to initialize the same global objects we normally
// do if this had been a Java binary.
void EnsureInitializedOnce() {
  RTC_CHECK(::webrtc::jni::GetJVM() != nullptr);
  JNIEnv* jni = ::webrtc::jni::AttachCurrentThreadIfNeeded();
  JavaVM* jvm = NULL;
  RTC_CHECK_EQ(0, jni->GetJavaVM(&jvm));

  RTC_CHECK(rtc::InitializeSSL()) << "Failed to InitializeSSL()";

  JVM::Initialize(jvm);
}

}  // anonymous namespace

void InitializeAndroidObjects() {
  RTC_CHECK_EQ(0, pthread_once(&g_initialize_once, &EnsureInitializedOnce));
}

}  // namespace webrtc
