/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 25, 2023.
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
 * TrustSettingsUtils.h - Utility routines for TrustSettings module
 */
 
#ifndef	_TRUST_SETTINGS_UTILS_H_
#define _TRUST_SETTINGS_UTILS_H_

#include <security_keychain/TrustSettings.h>
#include <Security/SecTrustSettingsPriv.h>
#include <security_utilities/alloc.h>
#include <string>
#include <CoreFoundation/CoreFoundation.h>

#define CFRELEASE(cf)		if(cf) { CFRelease(cf); }

#define TS_REQUIRED(arg)	if(arg == NULL) { return errSecParam; }

namespace Security
{

namespace KeychainCore
{

/* Read entire file. */
int tsReadFile(
	const char		*fileName,
	Allocator		&alloc,
	CSSM_DATA		&fileData);		// mallocd via alloc and RETURNED

} /* end namespace KeychainCore */

} /* end namespace Security */

#endif	/* _TRUST_SETTINGS_UTILS_H_ */
