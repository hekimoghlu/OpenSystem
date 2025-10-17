/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 6, 2022.
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
/*
 * Regress test for sshbuf.h buffer API
 *
 * Placed in the public domain
 */

#include "../test_helper/test_helper.h"

void sshbuf_tests(void);
void sshbuf_getput_basic_tests(void);
void sshbuf_getput_crypto_tests(void);
void sshbuf_misc_tests(void);
void sshbuf_fuzz_tests(void);
void sshbuf_getput_fuzz_tests(void);
void sshbuf_fixed(void);

void
tests(void)
{
	sshbuf_tests();
	sshbuf_getput_basic_tests();
#ifdef WITH_OPENSSL
	sshbuf_getput_crypto_tests();
#endif
	sshbuf_misc_tests();
	sshbuf_fuzz_tests();
	sshbuf_getput_fuzz_tests();
	sshbuf_fixed();
}
