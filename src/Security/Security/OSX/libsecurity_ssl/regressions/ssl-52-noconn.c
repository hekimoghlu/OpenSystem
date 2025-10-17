/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 27, 2023.
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

//
//  ssl-52-noconn.c
//  libsecurity_ssl
//

#include <stdio.h>
#include <Security/SecureTransport.h>
#include "ssl_regressions.h"

static
OSStatus r(SSLConnectionRef connection, void *data, size_t *dataLength) {
    return errSSLWouldBlock;
}

static
OSStatus w(SSLConnectionRef connection, const void *data, size_t *dataLength) {
    return errSSLWouldBlock;
}

//Testing <rdar://problem/13539215> Trivial SecureTransport example crashes on Cab, where it worked on Zin
static
void tests(void)
{
    OSStatus ortn;
    SSLContextRef ctx;
    ctx = SSLCreateContext(NULL, kSSLClientSide, kSSLStreamType);
    SSLSetIOFuncs(ctx, r, w);
    ortn = SSLHandshake(ctx);

    is(ortn, errSSLWouldBlock, "SSLHandshake unexpected return\n");

    CFRelease(ctx);
}


int ssl_52_noconn(int argc, char *const *argv)
{

    plan_tests(1);

    tests();

    return 0;
}
