/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 16, 2023.
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
#include "api/dtmf_sender_interface.h"
#include "sdk/android/generated_peerconnection_jni/DtmfSender_jni.h"
#include "sdk/android/native_api/jni/java_types.h"
#include "sdk/android/src/jni/jni_helpers.h"

namespace webrtc {
namespace jni {

static jboolean JNI_DtmfSender_CanInsertDtmf(JNIEnv* jni,
                                             jlong j_dtmf_sender_pointer) {
  return reinterpret_cast<DtmfSenderInterface*>(j_dtmf_sender_pointer)
      ->CanInsertDtmf();
}

static jboolean JNI_DtmfSender_InsertDtmf(JNIEnv* jni,
                                          jlong j_dtmf_sender_pointer,
                                          const JavaParamRef<jstring>& tones,
                                          jint duration,
                                          jint inter_tone_gap) {
  return reinterpret_cast<DtmfSenderInterface*>(j_dtmf_sender_pointer)
      ->InsertDtmf(JavaToStdString(jni, tones), duration, inter_tone_gap);
}

static ScopedJavaLocalRef<jstring> JNI_DtmfSender_Tones(
    JNIEnv* jni,
    jlong j_dtmf_sender_pointer) {
  return NativeToJavaString(
      jni,
      reinterpret_cast<DtmfSenderInterface*>(j_dtmf_sender_pointer)->tones());
}

static jint JNI_DtmfSender_Duration(JNIEnv* jni, jlong j_dtmf_sender_pointer) {
  return reinterpret_cast<DtmfSenderInterface*>(j_dtmf_sender_pointer)
      ->duration();
}

static jint JNI_DtmfSender_InterToneGap(JNIEnv* jni,
                                        jlong j_dtmf_sender_pointer) {
  return reinterpret_cast<DtmfSenderInterface*>(j_dtmf_sender_pointer)
      ->inter_tone_gap();
}

}  // namespace jni
}  // namespace webrtc
