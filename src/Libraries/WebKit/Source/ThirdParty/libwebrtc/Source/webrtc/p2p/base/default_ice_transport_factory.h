/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 5, 2025.
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
#ifndef P2P_BASE_DEFAULT_ICE_TRANSPORT_FACTORY_H_
#define P2P_BASE_DEFAULT_ICE_TRANSPORT_FACTORY_H_

#include <memory>
#include <string>

#include "api/ice_transport_interface.h"
#include "p2p/base/p2p_transport_channel.h"
#include "rtc_base/thread.h"

namespace webrtc {

// The default ICE transport wraps the implementation of IceTransportInternal
// provided by P2PTransportChannel. This default transport is not thread safe
// and must be constructed, used and destroyed on the same network thread on
// which the internal P2PTransportChannel lives.
class DefaultIceTransport : public IceTransportInterface {
 public:
  explicit DefaultIceTransport(
      std::unique_ptr<cricket::P2PTransportChannel> internal);
  ~DefaultIceTransport();

  cricket::IceTransportInternal* internal() override {
    RTC_DCHECK_RUN_ON(&thread_checker_);
    return internal_.get();
  }

 private:
  const SequenceChecker thread_checker_{};
  std::unique_ptr<cricket::P2PTransportChannel> internal_
      RTC_GUARDED_BY(thread_checker_);
};

class DefaultIceTransportFactory : public IceTransportFactory {
 public:
  DefaultIceTransportFactory() = default;
  ~DefaultIceTransportFactory() = default;

  // Must be called on the network thread and returns a DefaultIceTransport.
  rtc::scoped_refptr<IceTransportInterface> CreateIceTransport(
      const std::string& transport_name,
      int component,
      IceTransportInit init) override;
};

}  // namespace webrtc

#endif  // P2P_BASE_DEFAULT_ICE_TRANSPORT_FACTORY_H_
