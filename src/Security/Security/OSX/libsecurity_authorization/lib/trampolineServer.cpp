/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 12, 2022.
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
// trampolineServer.cpp - tool-side trampoline support functions
//
#include <cstdlib>
#include <unistd.h>
#include <Security/Authorization.h>
#include <Security/SecBase.h>
#include <dispatch/dispatch.h>
#include <security_utilities/debugging.h>

//
// In a tool launched via AuthorizationCopyPrivilegedReference, retrieve a copy
// of the AuthorizationRef that started it all.
//
OSStatus AuthorizationCopyPrivilegedReference(AuthorizationRef *authorization,
	AuthorizationFlags flags)
{
	secalert("AuthorizationCopyPrivilegedReference is deprecated and functionality will be removed in macOS 10.14 - please update your application");
	// flags are currently reserved
	if (flags != 0)
		return errAuthorizationInvalidFlags;

	// retrieve hex form of external form from environment
	const char *mboxFdText = getenv("__AUTHORIZATION");
	if (!mboxFdText) {
		return errAuthorizationInvalidRef;
	}

	static AuthorizationExternalForm extForm;
	static OSStatus result = errAuthorizationInvalidRef;
	static dispatch_once_t onceToken;
	dispatch_once(&onceToken, ^{
		// retrieve the pipe and read external form
		int fd;
		if (sscanf(mboxFdText, "auth %d", &fd) != 1) {
			return;
		}
		ssize_t numOfBytes = read(fd, &extForm, sizeof(extForm));
		close(fd);
		if (numOfBytes == sizeof(extForm)) {
			result = errAuthorizationSuccess;
		}
	});

	if (result) {
		// we had some trouble with reading the extform
		return result;
	}

	// internalize the authorization
	AuthorizationRef auth;
	if (OSStatus error = AuthorizationCreateFromExternalForm(&extForm, &auth))
		return error;

	if (authorization) {
		*authorization = auth;
	}

	return errAuthorizationSuccess;
}
