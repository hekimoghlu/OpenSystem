/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 12, 2023.
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
#include "rtc_base/internal/default_socket_server.h"

#include <memory>

#include "rtc_base/socket_server.h"

#if defined(__native_client__)
#include "rtc_base/null_socket_server.h"
#else
#include "rtc_base/physical_socket_server.h"
#endif

namespace rtc {

std::unique_ptr<SocketServer> CreateDefaultSocketServer() {
#if defined(__native_client__)
  return std::unique_ptr<SocketServer>(new rtc::NullSocketServer);
#else
  return std::unique_ptr<SocketServer>(new rtc::PhysicalSocketServer);
#endif
}

}  // namespace rtc
