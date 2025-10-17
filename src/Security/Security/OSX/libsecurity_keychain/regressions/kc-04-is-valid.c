/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 13, 2022.
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

#include <Security/SecKeychain.h>
#include <Security/SecKeychainPriv.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "keychain_regressions.h"
#include "kc-helpers.h"

static void tests(void)
{
	char *home = getenv("HOME");
	char kcname1[256], kcname2[256];

	if (!home || strlen(home) > 200)
		plan_skip_all("home too big");

	snprintf(kcname1,sizeof(kcname1), "%s/kctests/kc1-16-is-valid", home);
	SecKeychainRef kc1 = NULL, kc2 = NULL;
	Boolean kc1valid, kc2valid;
    kc1 = createNewKeychainAt(kcname1, "test");
	ok_status(SecKeychainIsValid(kc1, &kc1valid), "SecKeychainIsValid kc1");
	is(kc1valid, TRUE, "is kc1 valid");

    ok_status(SecKeychainDelete(kc1), "%s: SecKeychainDelete", testName);
	CFRelease(kc1);

	int fd;
	snprintf(kcname2,sizeof(kcname2), "%s/kctests/kc2-16-is-valid", home);
	ok_unix(fd = open(kcname2, O_CREAT|O_WRONLY|O_TRUNC, 0600),
		"create invalid kc2 file");
	ok_unix(close(fd), "close the kc2 file");
	ok_status(SecKeychainOpen(kcname2, &kc2), "SecKeychainOpen kc2");

	ok_status(SecKeychainIsValid(kc2, &kc2valid), "SecKeychainIsValid kc2");
	TODO: {
		todo("<rdar://problem/3795566> SecKeychainIsValid always returns "	
			"TRUE");
		is(kc2valid, FALSE, "is kc2 not valid");
	}

    ok_status(SecKeychainDelete(kc2), "%s: SecKeychainDelete", testName);
	CFRelease(kc2);
}

int kc_04_is_valid(int argc, char *const *argv)
{
	plan_tests(11);

	tests();

    deleteTestFiles();
	return 0;
}
