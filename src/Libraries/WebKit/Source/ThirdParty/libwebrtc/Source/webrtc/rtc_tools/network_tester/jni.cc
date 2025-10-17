/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 2, 2025.
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
#include <jni.h>
#undef JNIEXPORT
#define JNIEXPORT __attribute__((visibility("default")))
#include <string>

#include "rtc_base/logging.h"
#include "rtc_base/thread.h"
#include "rtc_tools/network_tester/test_controller.h"

extern "C" JNIEXPORT jlong JNICALL
Java_com_google_media_networktester_NetworkTester_CreateTestController(
    JNIEnv* jni,
    jclass) {
  rtc::ThreadManager::Instance()->WrapCurrentThread();
  return reinterpret_cast<intptr_t>(new webrtc::TestController(
      0, 0, "/mnt/sdcard/network_tester_client_config.dat",
      "/mnt/sdcard/network_tester_client_packet_log.dat"));
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_media_networktester_NetworkTester_TestControllerConnect(
    JNIEnv* jni,
    jclass,
    jlong native_pointer) {
  reinterpret_cast<webrtc::TestController*>(native_pointer)
      ->SendConnectTo("85.195.237.107", 9090);
}

extern "C" JNIEXPORT bool JNICALL
Java_com_google_media_networktester_NetworkTester_TestControllerIsDone(
    JNIEnv* jni,
    jclass,
    jlong native_pointer) {
  return reinterpret_cast<webrtc::TestController*>(native_pointer)
      ->IsTestDone();
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_media_networktester_NetworkTester_TestControllerRun(
    JNIEnv* jni,
    jclass,
    jlong native_pointer) {
  // 100 ms arbitrary chosen, but it works well.
  rtc::Thread::Current()->ProcessMessages(/*cms=*/100);
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_media_networktester_NetworkTester_DestroyTestController(
    JNIEnv* jni,
    jclass,
    jlong native_pointer) {
  webrtc::TestController* test_controller =
      reinterpret_cast<webrtc::TestController*>(native_pointer);
  if (test_controller) {
    delete test_controller;
  }
  rtc::ThreadManager::Instance()->UnwrapCurrentThread();
}
