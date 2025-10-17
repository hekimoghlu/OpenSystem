/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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
 *  AZNTest.cpp
 *  SecurityServer
 *
 *  Created by michael on Fri Oct 20 2000.
 */

#include <Security/Authorization.h>

#include <Security/AuthorizationEngine.h>

using namespace Authorization;

static const AuthorizationItem gItems[] =
{
	{"login", 0, NULL, NULL},
	{"reboot", 0, NULL, NULL},
	{"shutdown", 0, NULL, NULL},
	{"mount", 0, NULL, NULL},
	{"login.reboot", 0, NULL, NULL},
	{"login.shutdown", 0, NULL, NULL},
	{"unmount", 0, NULL, NULL}
};

static const AuthorizationRights gRights =
{
	7,
	const_cast<AuthorizationItem *>(gItems)
};

void
printRights(const RightSet &rightSet)
{
	for(RightSet::const_iterator it = rightSet.begin(); it != rightSet.end(); ++it)
	{
		printf("right: \"%s\"\n", it->rightName());
	}
}

int
main(int argc, char **argv)
{
	Engine engine("/tmp/config.plist");

	const RightSet inputRights(&gRights);
	MutableRightSet outputRights;
	printf("InputRights:\n");
	printRights(inputRights);
	printf("Authorizing:\n");
	OSStatus result = engine.authorize(inputRights, NULL,
		kAuthorizationFlagInteractionAllowed | kAuthorizationFlagExtendRights | kAuthorizationFlagPartialRights,
		NULL, NULL, &outputRights);
	printf("Result: %ld\n", result);
	printf("OutputRights:\n");
	printRights(outputRights);
	return 0;
}
