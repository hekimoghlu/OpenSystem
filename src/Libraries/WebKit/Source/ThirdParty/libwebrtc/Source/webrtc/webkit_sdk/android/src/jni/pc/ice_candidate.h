/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 10, 2022.
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
#ifndef SDK_ANDROID_SRC_JNI_PC_ICE_CANDIDATE_H_
#define SDK_ANDROID_SRC_JNI_PC_ICE_CANDIDATE_H_

#include <vector>

#include "api/data_channel_interface.h"
#include "api/jsep.h"
#include "api/jsep_ice_candidate.h"
#include "api/peer_connection_interface.h"
#include "api/rtp_parameters.h"
#include "rtc_base/ssl_identity.h"
#include "sdk/android/src/jni/jni_helpers.h"

namespace webrtc {
namespace jni {

cricket::Candidate JavaToNativeCandidate(JNIEnv* jni,
                                         const JavaRef<jobject>& j_candidate);

ScopedJavaLocalRef<jobject> NativeToJavaCandidate(
    JNIEnv* env,
    const cricket::Candidate& candidate);

ScopedJavaLocalRef<jobject> NativeToJavaIceCandidate(
    JNIEnv* env,
    const IceCandidateInterface& candidate);

ScopedJavaLocalRef<jobjectArray> NativeToJavaCandidateArray(
    JNIEnv* jni,
    const std::vector<cricket::Candidate>& candidates);

/*****************************************************
 * Below are all things that go into RTCConfiguration.
 *****************************************************/
PeerConnectionInterface::IceTransportsType JavaToNativeIceTransportsType(
    JNIEnv* jni,
    const JavaRef<jobject>& j_ice_transports_type);

PeerConnectionInterface::BundlePolicy JavaToNativeBundlePolicy(
    JNIEnv* jni,
    const JavaRef<jobject>& j_bundle_policy);

PeerConnectionInterface::RtcpMuxPolicy JavaToNativeRtcpMuxPolicy(
    JNIEnv* jni,
    const JavaRef<jobject>& j_rtcp_mux_policy);

PeerConnectionInterface::TcpCandidatePolicy JavaToNativeTcpCandidatePolicy(
    JNIEnv* jni,
    const JavaRef<jobject>& j_tcp_candidate_policy);

PeerConnectionInterface::CandidateNetworkPolicy
JavaToNativeCandidateNetworkPolicy(
    JNIEnv* jni,
    const JavaRef<jobject>& j_candidate_network_policy);

rtc::KeyType JavaToNativeKeyType(JNIEnv* jni,
                                 const JavaRef<jobject>& j_key_type);

PeerConnectionInterface::ContinualGatheringPolicy
JavaToNativeContinualGatheringPolicy(
    JNIEnv* jni,
    const JavaRef<jobject>& j_gathering_policy);

webrtc::PortPrunePolicy JavaToNativePortPrunePolicy(
    JNIEnv* jni,
    const JavaRef<jobject>& j_port_prune_policy);

PeerConnectionInterface::TlsCertPolicy JavaToNativeTlsCertPolicy(
    JNIEnv* jni,
    const JavaRef<jobject>& j_ice_server_tls_cert_policy);

absl::optional<rtc::AdapterType> JavaToNativeNetworkPreference(
    JNIEnv* jni,
    const JavaRef<jobject>& j_network_preference);

}  // namespace jni
}  // namespace webrtc

#endif  // SDK_ANDROID_SRC_JNI_PC_ICE_CANDIDATE_H_
