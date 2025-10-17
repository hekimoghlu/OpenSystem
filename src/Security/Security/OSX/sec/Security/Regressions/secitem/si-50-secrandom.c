/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 8, 2023.
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
#include <Security/SecRandom.h>
#include <CoreFoundation/CoreFoundation.h>
#include <stdlib.h>
#include <unistd.h>

#include "Security_regressions.h"

static int dummy;
static SecRandomRef kSecRandomNotDefault = (SecRandomRef)&dummy;

/* Test basic add delete update copy matching stuff. */
static void tests(void)
{
	UInt8 bytes[4096] = {};
	CFIndex size = 42;
	UInt8 *p = bytes + 23;
	ok_status(SecRandomCopyBytes(kSecRandomDefault, size, p), "generate some random bytes");
    ok_status(SecRandomCopyBytes(kSecRandomNotDefault, size, p), "ignore random implementation specifier");
}

int si_50_secrandom(int argc, char *const *argv)
{
	plan_tests(2);


	tests();

	return 0;
}
