/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 19, 2025.
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
#include "sdk/android/src/jni/pc/crypto_options.h"

#include "sdk/android/generated_peerconnection_jni/CryptoOptions_jni.h"

namespace webrtc {
namespace jni {

absl::optional<CryptoOptions> JavaToNativeOptionalCryptoOptions(
    JNIEnv* jni,
    const JavaRef<jobject>& j_crypto_options) {
  if (j_crypto_options.is_null()) {
    return absl::nullopt;
  }

  ScopedJavaLocalRef<jobject> j_srtp =
      Java_CryptoOptions_getSrtp(jni, j_crypto_options);
  ScopedJavaLocalRef<jobject> j_sframe =
      Java_CryptoOptions_getSFrame(jni, j_crypto_options);

  CryptoOptions native_crypto_options;
  native_crypto_options.srtp.enable_gcm_crypto_suites =
      Java_Srtp_getEnableGcmCryptoSuites(jni, j_srtp);
  native_crypto_options.srtp.enable_aes128_sha1_32_crypto_cipher =
      Java_Srtp_getEnableAes128Sha1_32CryptoCipher(jni, j_srtp);
  native_crypto_options.srtp.enable_encrypted_rtp_header_extensions =
      Java_Srtp_getEnableEncryptedRtpHeaderExtensions(jni, j_srtp);
  native_crypto_options.sframe.require_frame_encryption =
      Java_SFrame_getRequireFrameEncryption(jni, j_sframe);
  return absl::optional<CryptoOptions>(native_crypto_options);
}

}  // namespace jni
}  // namespace webrtc
