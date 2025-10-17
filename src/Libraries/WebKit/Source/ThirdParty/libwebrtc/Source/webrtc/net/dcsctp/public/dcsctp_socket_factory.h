/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 9, 2025.
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
#ifndef NET_DCSCTP_PUBLIC_DCSCTP_SOCKET_FACTORY_H_
#define NET_DCSCTP_PUBLIC_DCSCTP_SOCKET_FACTORY_H_

#include <memory>

#include "absl/strings/string_view.h"
#include "net/dcsctp/public/dcsctp_options.h"
#include "net/dcsctp/public/dcsctp_socket.h"
#include "net/dcsctp/public/packet_observer.h"

namespace dcsctp {
class DcSctpSocketFactory {
 public:
  virtual ~DcSctpSocketFactory();
  virtual std::unique_ptr<DcSctpSocketInterface> Create(
      absl::string_view log_prefix,
      DcSctpSocketCallbacks& callbacks,
      std::unique_ptr<PacketObserver> packet_observer,
      const DcSctpOptions& options);
};
}  // namespace dcsctp

#endif  // NET_DCSCTP_PUBLIC_DCSCTP_SOCKET_FACTORY_H_
