/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 29, 2024.
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
#include "api/peer_connection_interface.h"

#include <memory>
#include <utility>

#include "api/media_types.h"
#include "api/rtc_error.h"
#include "api/rtp_parameters.h"
#include "api/scoped_refptr.h"
#include "p2p/base/port_allocator.h"
#include "pc/media_factory.h"
#include "rtc_base/rtc_certificate_generator.h"

namespace webrtc {

PeerConnectionInterface::IceServer::IceServer() = default;
PeerConnectionInterface::IceServer::IceServer(const IceServer& rhs) = default;
PeerConnectionInterface::IceServer::~IceServer() = default;

PeerConnectionInterface::RTCConfiguration::RTCConfiguration() = default;

PeerConnectionInterface::RTCConfiguration::RTCConfiguration(
    const RTCConfiguration& rhs) = default;

PeerConnectionInterface::RTCConfiguration::RTCConfiguration(
    RTCConfigurationType type) {
  if (type == RTCConfigurationType::kAggressive) {
    // These parameters are also defined in Java and IOS configurations,
    // so their values may be overwritten by the Java or IOS configuration.
    bundle_policy = kBundlePolicyMaxBundle;
    rtcp_mux_policy = kRtcpMuxPolicyRequire;
    ice_connection_receiving_timeout = kAggressiveIceConnectionReceivingTimeout;

    // These parameters are not defined in Java or IOS configuration,
    // so their values will not be overwritten.
    enable_ice_renomination = true;
    redetermine_role_on_ice_restart = false;
  }
}

PeerConnectionInterface::RTCConfiguration::~RTCConfiguration() = default;

PeerConnectionDependencies::PeerConnectionDependencies(
    PeerConnectionObserver* observer_in)
    : observer(observer_in) {}

// TODO(bugs.webrtc.org/12598: remove pragma once async_resolver_factory
// is removed from PeerConnectionDependencies
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
PeerConnectionDependencies::PeerConnectionDependencies(
    PeerConnectionDependencies&&) = default;
#pragma clang diagnostic pop

PeerConnectionDependencies::~PeerConnectionDependencies() = default;

PeerConnectionFactoryDependencies::PeerConnectionFactoryDependencies() =
    default;

PeerConnectionFactoryDependencies::PeerConnectionFactoryDependencies(
    PeerConnectionFactoryDependencies&&) = default;

PeerConnectionFactoryDependencies::~PeerConnectionFactoryDependencies() =
    default;

rtc::scoped_refptr<PeerConnectionInterface>
PeerConnectionFactoryInterface::CreatePeerConnection(
    const PeerConnectionInterface::RTCConfiguration& configuration,
    std::unique_ptr<cricket::PortAllocator> allocator,
    std::unique_ptr<rtc::RTCCertificateGeneratorInterface> cert_generator,
    PeerConnectionObserver* observer) {
  PeerConnectionDependencies dependencies(observer);
  dependencies.allocator = std::move(allocator);
  dependencies.cert_generator = std::move(cert_generator);
  auto result =
      CreatePeerConnectionOrError(configuration, std::move(dependencies));
  if (!result.ok()) {
    return nullptr;
  }
  return result.MoveValue();
}

rtc::scoped_refptr<PeerConnectionInterface>
PeerConnectionFactoryInterface::CreatePeerConnection(
    const PeerConnectionInterface::RTCConfiguration& configuration,
    PeerConnectionDependencies dependencies) {
  auto result =
      CreatePeerConnectionOrError(configuration, std::move(dependencies));
  if (!result.ok()) {
    return nullptr;
  }
  return result.MoveValue();
}

RTCErrorOr<rtc::scoped_refptr<PeerConnectionInterface>>
PeerConnectionFactoryInterface::CreatePeerConnectionOrError(
    const PeerConnectionInterface::RTCConfiguration& /* configuration */,
    PeerConnectionDependencies /* dependencies */) {
  return RTCError(RTCErrorType::INTERNAL_ERROR);
}

RtpCapabilities PeerConnectionFactoryInterface::GetRtpSenderCapabilities(
    cricket::MediaType /* kind */) const {
  return {};
}

RtpCapabilities PeerConnectionFactoryInterface::GetRtpReceiverCapabilities(
    cricket::MediaType /* kind */) const {
  return {};
}

}  // namespace webrtc
