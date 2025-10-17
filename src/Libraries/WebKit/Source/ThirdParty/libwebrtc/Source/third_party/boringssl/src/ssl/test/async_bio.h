/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 30, 2025.
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
#ifndef HEADER_ASYNC_BIO
#define HEADER_ASYNC_BIO

#include <openssl/bio.h>


// AsyncBioCreate creates a filter BIO for testing asynchronous state
// machines which consume a stream socket. Reads and writes will fail
// and return EAGAIN unless explicitly allowed. Each async BIO has a
// read quota and a write quota. Initially both are zero. As each is
// incremented, bytes are allowed to flow through the BIO.
bssl::UniquePtr<BIO> AsyncBioCreate();

// AsyncBioCreateDatagram creates a filter BIO for testing for
// asynchronous state machines which consume datagram sockets. The read
// and write quota count in packets rather than bytes.
bssl::UniquePtr<BIO> AsyncBioCreateDatagram();

// AsyncBioAllowRead increments |bio|'s read quota by |count|.
void AsyncBioAllowRead(BIO *bio, size_t count);

// AsyncBioAllowWrite increments |bio|'s write quota by |count|.
void AsyncBioAllowWrite(BIO *bio, size_t count);

// AsyncBioEnforceWriteQuota configures where |bio| enforces its write quota.
void AsyncBioEnforceWriteQuota(BIO *bio, bool enforce);


#endif  // HEADER_ASYNC_BIO
