/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 17, 2025.
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
// SecCodeHost - Host Code API
//
#include "cs.h"
#include "SecCodeHost.h"
#include <security_utilities/cfutilities.h>
#include <security_utilities/globalizer.h>
#include <securityd_client/ssclient.h>

using namespace CodeSigning;


OSStatus SecHostCreateGuest(SecGuestRef host,
	uint32_t status, CFURLRef path, CFDictionaryRef attributes,
	SecCSFlags flags, SecGuestRef *newGuest)
{
	BEGIN_CSAPI
	
	MacOSError::throwMe(errSecCSNotSupported);
	
	END_CSAPI
}

OSStatus SecHostRemoveGuest(SecGuestRef host, SecGuestRef guest, SecCSFlags flags)
{
	BEGIN_CSAPI

	MacOSError::throwMe(errSecCSNotSupported);

	END_CSAPI
}

OSStatus SecHostSelectGuest(SecGuestRef guestRef, SecCSFlags flags)
{
	BEGIN_CSAPI

	MacOSError::throwMe(errSecCSNotSupported);

	END_CSAPI
}


OSStatus SecHostSelectedGuest(SecCSFlags flags, SecGuestRef *guestRef)
{
	BEGIN_CSAPI
	
	MacOSError::throwMe(errSecCSNotSupported);

	END_CSAPI
}

OSStatus SecHostSetGuestStatus(SecGuestRef guestRef,
	uint32_t status, CFDictionaryRef attributes,
	SecCSFlags flags)
{
	BEGIN_CSAPI

	MacOSError::throwMe(errSecCSNotSupported);

	END_CSAPI
}

OSStatus SecHostSetHostingPort(mach_port_t hostingPort, SecCSFlags flags)
{
	BEGIN_CSAPI

	MacOSError::throwMe(errSecCSNotSupported);

	END_CSAPI
}
