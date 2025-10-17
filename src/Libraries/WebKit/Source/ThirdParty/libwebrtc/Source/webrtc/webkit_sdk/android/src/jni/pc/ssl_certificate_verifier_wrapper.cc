/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 29, 2025.
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
#include "sdk/android/src/jni/pc/ssl_certificate_verifier_wrapper.h"

#include "sdk/android/generated_peerconnection_jni/SSLCertificateVerifier_jni.h"
#include "sdk/android/native_api/jni/class_loader.h"
#include "sdk/android/native_api/jni/java_types.h"

namespace webrtc {
namespace jni {

SSLCertificateVerifierWrapper::SSLCertificateVerifierWrapper(
    JNIEnv* jni,
    const JavaRef<jobject>& ssl_certificate_verifier)
    : ssl_certificate_verifier_(jni, ssl_certificate_verifier) {}

SSLCertificateVerifierWrapper::~SSLCertificateVerifierWrapper() = default;

bool SSLCertificateVerifierWrapper::Verify(
    const rtc::SSLCertificate& certificate) {
  JNIEnv* jni = AttachCurrentThreadIfNeeded();

  // Serialize the der encoding of the cert into a jbyteArray
  rtc::Buffer cert_der_buffer;
  certificate.ToDER(&cert_der_buffer);
  ScopedJavaLocalRef<jbyteArray> jni_buffer(
      jni, jni->NewByteArray(cert_der_buffer.size()));
  jni->SetByteArrayRegion(
      jni_buffer.obj(), 0, cert_der_buffer.size(),
      reinterpret_cast<const jbyte*>(cert_der_buffer.data()));

  return Java_SSLCertificateVerifier_verify(jni, ssl_certificate_verifier_,
                                            jni_buffer);
}

}  // namespace jni
}  // namespace webrtc
