/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 9, 2022.
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
#include "sdk/android/src/jni/pc/session_description.h"

#include <string>

#include "rtc_base/logging.h"
#include "sdk/android/generated_peerconnection_jni/SessionDescription_jni.h"
#include "sdk/android/native_api/jni/java_types.h"
#include "sdk/android/src/jni/jni_helpers.h"

namespace webrtc {
namespace jni {

std::unique_ptr<SessionDescriptionInterface> JavaToNativeSessionDescription(
    JNIEnv* jni,
    const JavaRef<jobject>& j_sdp) {
  std::string std_type = JavaToStdString(
      jni, Java_SessionDescription_getTypeInCanonicalForm(jni, j_sdp));
  std::string std_description =
      JavaToStdString(jni, Java_SessionDescription_getDescription(jni, j_sdp));
  absl::optional<SdpType> sdp_type_maybe = SdpTypeFromString(std_type);
  if (!sdp_type_maybe) {
    RTC_LOG(LS_ERROR) << "Unexpected SDP type: " << std_type;
    return nullptr;
  }
  return CreateSessionDescription(*sdp_type_maybe, std_description);
}

ScopedJavaLocalRef<jobject> NativeToJavaSessionDescription(
    JNIEnv* jni,
    const std::string& sdp,
    const std::string& type) {
  return Java_SessionDescription_Constructor(
      jni, Java_Type_fromCanonicalForm(jni, NativeToJavaString(jni, type)),
      NativeToJavaString(jni, sdp));
}

}  // namespace jni
}  // namespace webrtc
