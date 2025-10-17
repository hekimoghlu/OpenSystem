/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 13, 2022.
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
// cs.h - code signing core header
//
#include "cs.h"
#include <security_utilities/cfmunge.h>

namespace Security {
namespace CodeSigning {


ModuleNexus<CFObjects> gCFObjects;

CFObjects::CFObjects()
	: Code("SecCode"),
	  StaticCode("SecStaticCode"),
	  Requirement("SecRequirements"),
	  CodeSigner("SecCodeSigner"),
      CodeSignerRemote("SecCodeSignerRemote")
{
}


OSStatus dbError(const SQLite3::Error &err)
{
	switch (err.error) {
	case SQLITE_PERM:
	case SQLITE_READONLY:
	case SQLITE_AUTH:
		return errSecCSSigDBDenied;
	case SQLITE_CANTOPEN:
	case SQLITE_EMPTY:
	case SQLITE_NOTADB:
		return errSecCSSigDBAccess;
	default:
		return SecKeychainErrFromOSStatus(err.osStatus());
	}
}


}	// CodeSigning
}	// Security
