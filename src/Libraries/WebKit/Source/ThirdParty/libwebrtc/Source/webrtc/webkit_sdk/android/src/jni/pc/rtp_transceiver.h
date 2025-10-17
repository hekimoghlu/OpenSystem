/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 18, 2024.
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
#ifndef SDK_ANDROID_SRC_JNI_PC_RTP_TRANSCEIVER_H_
#define SDK_ANDROID_SRC_JNI_PC_RTP_TRANSCEIVER_H_

#include <jni.h>

#include "api/rtp_transceiver_interface.h"
#include "sdk/android/native_api/jni/scoped_java_ref.h"

namespace webrtc {
namespace jni {

RtpTransceiverInit JavaToNativeRtpTransceiverInit(
    JNIEnv* jni,
    const JavaRef<jobject>& j_init);

ScopedJavaLocalRef<jobject> NativeToJavaRtpTransceiver(
    JNIEnv* env,
    rtc::scoped_refptr<RtpTransceiverInterface> transceiver);

// This takes ownership of the of the `j_transceiver` and stores it as a global
// reference. This calls the Java Transceiver's dispose() method with the dtor.
class JavaRtpTransceiverGlobalOwner {
 public:
  JavaRtpTransceiverGlobalOwner(JNIEnv* env,
                                const JavaRef<jobject>& j_transceiver);
  JavaRtpTransceiverGlobalOwner(JavaRtpTransceiverGlobalOwner&& other);
  ~JavaRtpTransceiverGlobalOwner();

 private:
  ScopedJavaGlobalRef<jobject> j_transceiver_;
};

}  // namespace jni
}  // namespace webrtc

#endif  // SDK_ANDROID_SRC_JNI_PC_RTP_TRANSCEIVER_H_
