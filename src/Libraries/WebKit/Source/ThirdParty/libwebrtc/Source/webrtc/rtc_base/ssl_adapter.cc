/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 9, 2023.
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
#include "rtc_base/ssl_adapter.h"

#include <memory>

#include "rtc_base/openssl_adapter.h"
#include "rtc_base/socket.h"

///////////////////////////////////////////////////////////////////////////////

namespace rtc {

std::unique_ptr<SSLAdapterFactory> SSLAdapterFactory::Create() {
  return std::make_unique<OpenSSLAdapterFactory>();
}

SSLAdapter* SSLAdapter::Create(Socket* socket) {
  return new OpenSSLAdapter(socket);
}

///////////////////////////////////////////////////////////////////////////////

bool InitializeSSL() {
  return OpenSSLAdapter::InitializeSSL();
}

bool CleanupSSL() {
  return OpenSSLAdapter::CleanupSSL();
}

///////////////////////////////////////////////////////////////////////////////

}  // namespace rtc
