/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 5, 2024.
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
#ifndef SDK_ANDROID_SRC_JNI_PC_SSL_CERTIFICATE_VERIFIER_WRAPPER_H_
#define SDK_ANDROID_SRC_JNI_PC_SSL_CERTIFICATE_VERIFIER_WRAPPER_H_

#include <jni.h>

#include <vector>

#include "rtc_base/ssl_certificate.h"
#include "sdk/android/src/jni/jni_helpers.h"

namespace webrtc {
namespace jni {

// Wrapper for Java SSLCertifiacteVerifier class. Delegates method calls through
// JNI and wraps the encoder inside SSLCertificateVerifierWrapper.
class SSLCertificateVerifierWrapper : public rtc::SSLCertificateVerifier {
 public:
  SSLCertificateVerifierWrapper(
      JNIEnv* jni,
      const JavaRef<jobject>& ssl_certificate_verifier);
  ~SSLCertificateVerifierWrapper() override;

  bool Verify(const rtc::SSLCertificate& certificate) override;

 private:
  const ScopedJavaGlobalRef<jobject> ssl_certificate_verifier_;
};

}  // namespace jni
}  // namespace webrtc

#endif  // SDK_ANDROID_SRC_JNI_PC_SSL_CERTIFICATE_VERIFIER_WRAPPER_H_
