/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 31, 2024.
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
#import "RTCConfiguration.h"

#include "api/peer_connection_interface.h"

NS_ASSUME_NONNULL_BEGIN

@interface RTCConfiguration ()

+ (webrtc::PeerConnectionInterface::IceTransportsType)nativeTransportsTypeForTransportPolicy:
        (RTCIceTransportPolicy)policy;

+ (RTCIceTransportPolicy)transportPolicyForTransportsType:
        (webrtc::PeerConnectionInterface::IceTransportsType)nativeType;

+ (NSString *)stringForTransportPolicy:(RTCIceTransportPolicy)policy;

+ (webrtc::PeerConnectionInterface::BundlePolicy)nativeBundlePolicyForPolicy:
        (RTCBundlePolicy)policy;

+ (RTCBundlePolicy)bundlePolicyForNativePolicy:
        (webrtc::PeerConnectionInterface::BundlePolicy)nativePolicy;

+ (NSString *)stringForBundlePolicy:(RTCBundlePolicy)policy;

+ (webrtc::PeerConnectionInterface::RtcpMuxPolicy)nativeRtcpMuxPolicyForPolicy:
        (RTCRtcpMuxPolicy)policy;

+ (RTCRtcpMuxPolicy)rtcpMuxPolicyForNativePolicy:
        (webrtc::PeerConnectionInterface::RtcpMuxPolicy)nativePolicy;

+ (NSString *)stringForRtcpMuxPolicy:(RTCRtcpMuxPolicy)policy;

+ (webrtc::PeerConnectionInterface::TcpCandidatePolicy)nativeTcpCandidatePolicyForPolicy:
        (RTCTcpCandidatePolicy)policy;

+ (RTCTcpCandidatePolicy)tcpCandidatePolicyForNativePolicy:
        (webrtc::PeerConnectionInterface::TcpCandidatePolicy)nativePolicy;

+ (NSString *)stringForTcpCandidatePolicy:(RTCTcpCandidatePolicy)policy;

+ (webrtc::PeerConnectionInterface::CandidateNetworkPolicy)nativeCandidateNetworkPolicyForPolicy:
        (RTCCandidateNetworkPolicy)policy;

+ (RTCCandidateNetworkPolicy)candidateNetworkPolicyForNativePolicy:
        (webrtc::PeerConnectionInterface::CandidateNetworkPolicy)nativePolicy;

+ (NSString *)stringForCandidateNetworkPolicy:(RTCCandidateNetworkPolicy)policy;

+ (rtc::KeyType)nativeEncryptionKeyTypeForKeyType:(RTCEncryptionKeyType)keyType;

+ (webrtc::SdpSemantics)nativeSdpSemanticsForSdpSemantics:(RTCSdpSemantics)sdpSemantics;

+ (RTCSdpSemantics)sdpSemanticsForNativeSdpSemantics:(webrtc::SdpSemantics)sdpSemantics;

+ (NSString *)stringForSdpSemantics:(RTCSdpSemantics)sdpSemantics;

/**
 * RTCConfiguration struct representation of this RTCConfiguration. This is
 * needed to pass to the underlying C++ APIs.
 */
- (nullable webrtc::PeerConnectionInterface::RTCConfiguration *)createNativeConfiguration;

- (instancetype)initWithNativeConfiguration:
        (const webrtc::PeerConnectionInterface::RTCConfiguration &)config NS_DESIGNATED_INITIALIZER;

@end

NS_ASSUME_NONNULL_END
