/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 10, 2022.
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
#ifndef SDK_ANDROID_SRC_JNI_PC_SDP_OBSERVER_H_
#define SDK_ANDROID_SRC_JNI_PC_SDP_OBSERVER_H_

#include <memory>
#include <string>

#include "api/peer_connection_interface.h"
#include "sdk/android/src/jni/jni_helpers.h"
#include "sdk/android/src/jni/pc/session_description.h"
#include "sdk/media_constraints.h"

namespace webrtc {
namespace jni {

class CreateSdpObserverJni : public CreateSessionDescriptionObserver {
 public:
  CreateSdpObserverJni(JNIEnv* env,
                       const JavaRef<jobject>& j_observer,
                       std::unique_ptr<MediaConstraints> constraints);
  ~CreateSdpObserverJni() override;

  MediaConstraints* constraints() { return constraints_.get(); }

  void OnSuccess(SessionDescriptionInterface* desc) override;
  void OnFailure(RTCError error) override;

 private:
  const ScopedJavaGlobalRef<jobject> j_observer_global_;
  std::unique_ptr<MediaConstraints> constraints_;
};

class SetLocalSdpObserverJni : public SetLocalDescriptionObserverInterface {
 public:
  SetLocalSdpObserverJni(JNIEnv* env, const JavaRef<jobject>& j_observer);

  ~SetLocalSdpObserverJni() override = default;

  virtual void OnSetLocalDescriptionComplete(RTCError error) override;

 private:
  const ScopedJavaGlobalRef<jobject> j_observer_global_;
};

class SetRemoteSdpObserverJni : public SetRemoteDescriptionObserverInterface {
 public:
  SetRemoteSdpObserverJni(JNIEnv* env, const JavaRef<jobject>& j_observer);

  ~SetRemoteSdpObserverJni() override = default;

  virtual void OnSetRemoteDescriptionComplete(RTCError error) override;

 private:
  const ScopedJavaGlobalRef<jobject> j_observer_global_;
};

}  // namespace jni
}  // namespace webrtc

#endif  // SDK_ANDROID_SRC_JNI_PC_SDP_OBSERVER_H_
