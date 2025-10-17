/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 7, 2024.
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
#ifndef HEADER_PACKETED_BIO
#define HEADER_PACKETED_BIO

#include <functional>

#include <openssl/base.h>
#include <openssl/bio.h>

#if defined(OPENSSL_WINDOWS)
OPENSSL_MSVC_PRAGMA(warning(push, 3))
#include <winsock2.h>
OPENSSL_MSVC_PRAGMA(warning(pop))
#else
#include <sys/time.h>
#endif


// PacketedBioCreate creates a filter BIO which implements a reliable in-order
// blocking datagram socket. It uses the value of |*clock| as the clock.
// |set_mtu| will be called when the runner asks to change the MTU.
//
// During a |BIO_read|, the peer may signal the filter BIO to simulate a
// timeout. The operation will fail immediately. The caller must then call
// |PacketedBioAdvanceClock| before retrying |BIO_read|.
bssl::UniquePtr<BIO> PacketedBioCreate(timeval *clock,
                                       std::function<bool(uint32_t)> set_mtu);

// PacketedBioAdvanceClock advances |bio|'s clock and returns true if there is a
// pending timeout. Otherwise, it returns false.
bool PacketedBioAdvanceClock(BIO *bio);


#endif  // HEADER_PACKETED_BIO
