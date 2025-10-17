/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 3, 2025.
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
//  Copyright (c) 2011 Apple. All rights reserved.
//

#include <Security/Security.h>
#include <Security/SecTask.h>
#include <stdio.h>
#include <err.h>

int main (int argc, const char * argv[])
{
	long num = 1000;

	while (num-- > 0) {
		SecTaskRef secTask = SecTaskCreateFromSelf(NULL);
		if (secTask == NULL)
			errx(1, "SecTaskCreateFromSelf");

		CFErrorRef error = NULL;
		CFTypeRef value = SecTaskCopyValueForEntitlement(secTask, CFSTR("com.apple.security.some-entitlement"), &error);
		if (value == NULL)
			errx(1, "SecTaskCopyValueForEntitlement");

		if (num == 1)
			CFShow(value);

		CFRelease(value);
		CFRelease(secTask);
	}

	return 0;
}

