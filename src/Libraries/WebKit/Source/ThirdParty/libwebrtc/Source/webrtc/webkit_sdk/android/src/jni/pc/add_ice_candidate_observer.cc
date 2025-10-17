/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 3, 2023.
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
#include "sdk/android/src/jni/pc/add_ice_candidate_observer.h"

#include <utility>

#include "sdk/android/generated_peerconnection_jni/AddIceObserver_jni.h"
#include "sdk/android/native_api/jni/java_types.h"
#include "sdk/android/src/jni/jni_helpers.h"
#include "sdk/media_constraints.h"

namespace webrtc {
namespace jni {

AddIceCandidateObserverJni::AddIceCandidateObserverJni(
    JNIEnv* env,
    const JavaRef<jobject>& j_observer)
    : j_observer_global_(env, j_observer) {}

void AddIceCandidateObserverJni::OnComplete(webrtc::RTCError error) {
  JNIEnv* env = AttachCurrentThreadIfNeeded();
  if (error.ok()) {
    Java_AddIceObserver_onAddSuccess(env, j_observer_global_);
  } else {
    Java_AddIceObserver_onAddFailure(env, j_observer_global_,
                                     NativeToJavaString(env, error.message()));
  }
}

}  // namespace jni
}  // namespace webrtc
