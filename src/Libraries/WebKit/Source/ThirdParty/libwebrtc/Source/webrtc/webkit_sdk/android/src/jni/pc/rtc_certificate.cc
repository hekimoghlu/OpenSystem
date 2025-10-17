/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 1, 2022.
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
#include "sdk/android/src/jni/pc/rtc_certificate.h"

#include "rtc_base/ref_count.h"
#include "rtc_base/rtc_certificate.h"
#include "rtc_base/rtc_certificate_generator.h"
#include "sdk/android/generated_peerconnection_jni/RtcCertificatePem_jni.h"
#include "sdk/android/native_api/jni/java_types.h"
#include "sdk/android/src/jni/jni_helpers.h"
#include "sdk/android/src/jni/pc/ice_candidate.h"

namespace webrtc {
namespace jni {

rtc::RTCCertificatePEM JavaToNativeRTCCertificatePEM(
    JNIEnv* jni,
    const JavaRef<jobject>& j_rtc_certificate) {
  ScopedJavaLocalRef<jstring> privatekey_field =
      Java_RtcCertificatePem_getPrivateKey(jni, j_rtc_certificate);
  ScopedJavaLocalRef<jstring> certificate_field =
      Java_RtcCertificatePem_getCertificate(jni, j_rtc_certificate);
  return rtc::RTCCertificatePEM(JavaToNativeString(jni, privatekey_field),
                                JavaToNativeString(jni, certificate_field));
}

ScopedJavaLocalRef<jobject> NativeToJavaRTCCertificatePEM(
    JNIEnv* jni,
    const rtc::RTCCertificatePEM& certificate) {
  return Java_RtcCertificatePem_Constructor(
      jni, NativeToJavaString(jni, certificate.private_key()),
      NativeToJavaString(jni, certificate.certificate()));
}

static ScopedJavaLocalRef<jobject> JNI_RtcCertificatePem_GenerateCertificate(
    JNIEnv* jni,
    const JavaParamRef<jobject>& j_key_type,
    jlong j_expires) {
  rtc::KeyType key_type = JavaToNativeKeyType(jni, j_key_type);
  uint64_t expires = (uint64_t)j_expires;
  rtc::scoped_refptr<rtc::RTCCertificate> certificate =
      rtc::RTCCertificateGenerator::GenerateCertificate(
          rtc::KeyParams(key_type), expires);
  rtc::RTCCertificatePEM pem = certificate->ToPEM();
  return Java_RtcCertificatePem_Constructor(
      jni, NativeToJavaString(jni, pem.private_key()),
      NativeToJavaString(jni, pem.certificate()));
}

}  // namespace jni
}  // namespace webrtc
