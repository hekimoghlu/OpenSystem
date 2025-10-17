/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 14, 2024.
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
#include "sdk/android/src/jni/pc/rtp_sender.h"

#include "sdk/android/generated_peerconnection_jni/RtpSender_jni.h"
#include "sdk/android/native_api/jni/java_types.h"
#include "sdk/android/src/jni/jni_helpers.h"
#include "sdk/android/src/jni/pc/rtp_parameters.h"

namespace webrtc {
namespace jni {

ScopedJavaLocalRef<jobject> NativeToJavaRtpSender(
    JNIEnv* env,
    rtc::scoped_refptr<RtpSenderInterface> sender) {
  if (!sender)
    return nullptr;
  // Sender is now owned by the Java object, and will be freed from
  // RtpSender.dispose(), called by PeerConnection.dispose() or getSenders().
  return Java_RtpSender_Constructor(env, jlongFromPointer(sender.release()));
}

static jboolean JNI_RtpSender_SetTrack(JNIEnv* jni,
                                       jlong j_rtp_sender_pointer,
                                       jlong j_track_pointer) {
  return reinterpret_cast<RtpSenderInterface*>(j_rtp_sender_pointer)
      ->SetTrack(reinterpret_cast<MediaStreamTrackInterface*>(j_track_pointer));
}

jlong JNI_RtpSender_GetTrack(JNIEnv* jni, jlong j_rtp_sender_pointer) {
  // MediaStreamTrack will have shared ownership by the MediaStreamTrack Java
  // object.
  return jlongFromPointer(
      reinterpret_cast<RtpSenderInterface*>(j_rtp_sender_pointer)
          ->track()
          .release());
}

static void JNI_RtpSender_SetStreams(
    JNIEnv* jni,
    jlong j_rtp_sender_pointer,
    const JavaParamRef<jobject>& j_stream_labels) {
  reinterpret_cast<RtpSenderInterface*>(j_rtp_sender_pointer)
      ->SetStreams(JavaListToNativeVector<std::string, jstring>(
          jni, j_stream_labels, &JavaToNativeString));
}

ScopedJavaLocalRef<jobject> JNI_RtpSender_GetStreams(
    JNIEnv* jni,
    jlong j_rtp_sender_pointer) {
  ScopedJavaLocalRef<jstring> (*convert_function)(JNIEnv*, const std::string&) =
      &NativeToJavaString;
  return NativeToJavaList(
      jni,
      reinterpret_cast<RtpSenderInterface*>(j_rtp_sender_pointer)->stream_ids(),
      convert_function);
}

jlong JNI_RtpSender_GetDtmfSender(JNIEnv* jni, jlong j_rtp_sender_pointer) {
  return jlongFromPointer(
      reinterpret_cast<RtpSenderInterface*>(j_rtp_sender_pointer)
          ->GetDtmfSender()
          .release());
}

jboolean JNI_RtpSender_SetParameters(
    JNIEnv* jni,
    jlong j_rtp_sender_pointer,
    const JavaParamRef<jobject>& j_parameters) {
  if (IsNull(jni, j_parameters)) {
    return false;
  }
  RtpParameters parameters = JavaToNativeRtpParameters(jni, j_parameters);
  return reinterpret_cast<RtpSenderInterface*>(j_rtp_sender_pointer)
      ->SetParameters(parameters)
      .ok();
}

ScopedJavaLocalRef<jobject> JNI_RtpSender_GetParameters(
    JNIEnv* jni,
    jlong j_rtp_sender_pointer) {
  RtpParameters parameters =
      reinterpret_cast<RtpSenderInterface*>(j_rtp_sender_pointer)
          ->GetParameters();
  return NativeToJavaRtpParameters(jni, parameters);
}

ScopedJavaLocalRef<jstring> JNI_RtpSender_GetId(JNIEnv* jni,
                                                jlong j_rtp_sender_pointer) {
  return NativeToJavaString(
      jni, reinterpret_cast<RtpSenderInterface*>(j_rtp_sender_pointer)->id());
}

static void JNI_RtpSender_SetFrameEncryptor(JNIEnv* jni,
                                            jlong j_rtp_sender_pointer,
                                            jlong j_frame_encryptor_pointer) {
  reinterpret_cast<RtpSenderInterface*>(j_rtp_sender_pointer)
      ->SetFrameEncryptor(rtc::scoped_refptr<FrameEncryptorInterface>(
          reinterpret_cast<FrameEncryptorInterface*>(
              j_frame_encryptor_pointer)));
}

static ScopedJavaLocalRef<jstring> JNI_RtpSender_GetMediaType(
    JNIEnv* jni,
    jlong j_rtp_sender_pointer) {
  cricket::MediaType media_type =
      reinterpret_cast<RtpSenderInterface*>(j_rtp_sender_pointer)->media_type();
  return media_type == cricket::MEDIA_TYPE_AUDIO
             ? NativeToJavaString(jni, "audio")
             : NativeToJavaString(jni, "video");
}

}  // namespace jni
}  // namespace webrtc
