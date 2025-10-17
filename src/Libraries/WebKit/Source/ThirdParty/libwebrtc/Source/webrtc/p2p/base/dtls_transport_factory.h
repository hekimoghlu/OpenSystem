/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 11, 2022.
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
#ifndef P2P_BASE_DTLS_TRANSPORT_FACTORY_H_
#define P2P_BASE_DTLS_TRANSPORT_FACTORY_H_

#include <memory>
#include <string>

#include "p2p/base/dtls_transport_internal.h"
#include "p2p/base/ice_transport_internal.h"

namespace cricket {

// This interface is used to create DTLS transports. The external transports
// can be injected into the JsepTransportController through it.
//
// TODO(qingsi): Remove this factory in favor of one that produces
// DtlsTransportInterface given by the public API if this is going to be
// injectable.
class DtlsTransportFactory {
 public:
  virtual ~DtlsTransportFactory() = default;

  virtual std::unique_ptr<DtlsTransportInternal> CreateDtlsTransport(
      IceTransportInternal* ice,
      const webrtc::CryptoOptions& crypto_options,
      rtc::SSLProtocolVersion max_version) = 0;
};

}  // namespace cricket

#endif  // P2P_BASE_DTLS_TRANSPORT_FACTORY_H_
