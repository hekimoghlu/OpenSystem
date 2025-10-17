/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 16, 2022.
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
#include "api/turn_customizer.h"

#include "sdk/android/generated_peerconnection_jni/TurnCustomizer_jni.h"
#include "sdk/android/native_api/jni/java_types.h"
#include "sdk/android/src/jni/jni_helpers.h"

namespace webrtc {
namespace jni {

TurnCustomizer* GetNativeTurnCustomizer(
    JNIEnv* env,
    const JavaRef<jobject>& j_turn_customizer) {
  if (IsNull(env, j_turn_customizer))
    return nullptr;
  return reinterpret_cast<webrtc::TurnCustomizer*>(
      Java_TurnCustomizer_getNativeTurnCustomizer(env, j_turn_customizer));
}

static void JNI_TurnCustomizer_FreeTurnCustomizer(
    JNIEnv* jni,
    jlong j_turn_customizer_pointer) {
  delete reinterpret_cast<TurnCustomizer*>(j_turn_customizer_pointer);
}

}  // namespace jni
}  // namespace webrtc
