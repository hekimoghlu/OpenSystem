/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 2, 2022.
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
#include "api/ice_transport_factory.h"

#include <memory>
#include <utility>

#include "api/ice_transport_interface.h"
#include "api/make_ref_counted.h"
#include "api/scoped_refptr.h"
#include "api/sequence_checker.h"
#include "p2p/base/ice_transport_internal.h"
#include "p2p/base/p2p_constants.h"
#include "p2p/base/p2p_transport_channel.h"
#include "p2p/base/port_allocator.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {

namespace {

// This implementation of IceTransportInterface is used in cases where
// the only reference to the P2PTransport will be through this class.
// It must be constructed, accessed and destroyed on the signaling thread.
class IceTransportWithTransportChannel : public IceTransportInterface {
 public:
  IceTransportWithTransportChannel(
      std::unique_ptr<cricket::IceTransportInternal> internal)
      : internal_(std::move(internal)) {}

  ~IceTransportWithTransportChannel() override {
    RTC_DCHECK_RUN_ON(&thread_checker_);
  }

  cricket::IceTransportInternal* internal() override {
    RTC_DCHECK_RUN_ON(&thread_checker_);
    return internal_.get();
  }

 private:
  const SequenceChecker thread_checker_{};
  const std::unique_ptr<cricket::IceTransportInternal> internal_
      RTC_GUARDED_BY(thread_checker_);
};

}  // namespace

rtc::scoped_refptr<IceTransportInterface> CreateIceTransport(
    cricket::PortAllocator* port_allocator) {
  IceTransportInit init;
  init.set_port_allocator(port_allocator);
  return CreateIceTransport(std::move(init));
}

rtc::scoped_refptr<IceTransportInterface> CreateIceTransport(
    IceTransportInit init) {
  return rtc::make_ref_counted<IceTransportWithTransportChannel>(
      cricket::P2PTransportChannel::Create(
          "", cricket::ICE_CANDIDATE_COMPONENT_RTP, std::move(init)));
}

}  // namespace webrtc
