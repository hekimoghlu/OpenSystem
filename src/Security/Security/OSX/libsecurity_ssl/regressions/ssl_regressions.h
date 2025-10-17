/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 30, 2024.
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

#include <regressions/test/testmore.h>


#if TARGET_OS_IPHONE

OFF_ONE_TEST(ssl_39_echo)
ONE_TEST(ssl_40_clientauth)
ONE_TEST(ssl_41_clientauth)

#else

DISABLED_ONE_TEST(ssl_39_echo)
DISABLED_ONE_TEST(ssl_40_clientauth)
DISABLED_ONE_TEST(ssl_41_clientauth)

#endif

ONE_TEST(ssl_42_ciphers)

OFF_ONE_TEST(ssl_43_ciphers)

ONE_TEST(ssl_44_crashes)
OFF_ONE_TEST(ssl_45_tls12)
ONE_TEST(ssl_46_SSLGetSupportedCiphers)
OFF_ONE_TEST(ssl_47_falsestart)
ONE_TEST(ssl_48_split)
// This one require a version of coreTLS that support SNI server side. (> coreTLS-17 ?)
OFF_ONE_TEST(ssl_49_sni)
OFF_ONE_TEST(ssl_50_server)

ONE_TEST(ssl_51_state)
ONE_TEST(ssl_52_noconn)
ONE_TEST(ssl_53_clientauth)
ONE_TEST(ssl_54_dhe)
ONE_TEST(ssl_55_sessioncache)
ONE_TEST(ssl_56_renegotiate)

