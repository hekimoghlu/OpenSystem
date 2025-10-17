/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 3, 2021.
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
#include "p2p/base/default_ice_transport_factory.h"

#include <utility>

#include "api/make_ref_counted.h"
#include "p2p/base/basic_ice_controller.h"
#include "p2p/base/ice_controller_factory_interface.h"

namespace {

class BasicIceControllerFactory
    : public cricket::IceControllerFactoryInterface {
 public:
  std::unique_ptr<cricket::IceControllerInterface> Create(
      const cricket::IceControllerFactoryArgs& args) override {
    return std::make_unique<cricket::BasicIceController>(args);
  }
};

}  // namespace

namespace webrtc {

DefaultIceTransport::DefaultIceTransport(
    std::unique_ptr<cricket::P2PTransportChannel> internal)
    : internal_(std::move(internal)) {}

DefaultIceTransport::~DefaultIceTransport() {
  RTC_DCHECK_RUN_ON(&thread_checker_);
}

rtc::scoped_refptr<IceTransportInterface>
DefaultIceTransportFactory::CreateIceTransport(
    const std::string& transport_name,
    int component,
    IceTransportInit init) {
  BasicIceControllerFactory factory;
  init.set_ice_controller_factory(&factory);
  return rtc::make_ref_counted<DefaultIceTransport>(
      cricket::P2PTransportChannel::Create(transport_name, component,
                                           std::move(init)));
}

}  // namespace webrtc
