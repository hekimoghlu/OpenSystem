/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 15, 2024.
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
 * CMSUtils.h - common utility routines for libCMS.
 */
 
#ifndef	_CMS_UTILS_H_
#define _CMS_UTILS_H_

#include <Security/cssmtype.h>
#include <Security/cssmapple.h>		/* cssmPerror() */
#include <CoreFoundation/CoreFoundation.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Copy a CSSM_DATA, mallocing the result.
 */
void cmsCopyCmsData(
	const CSSM_DATA *src,
	CSSM_DATA *dst);
	
/* 
 * Append a CF type, or the contents of an array, to another array.
 * destination array will be created if necessary.
 * If srcItemOrArray is not of the type specified in expectedType,
 * errSecParam will be returned. 
 */
OSStatus cmsAppendToArray(
	CFTypeRef srcItemOrArray,
	CFMutableArrayRef *dstArray,
	CFTypeID expectedType);

/* 
 * Munge an OSStatus returned from libsecurity_smime, which may well be an ASN.1 private
 * error code, to a real OSStatus.
 */
OSStatus cmsRtnToOSStatus(
	OSStatus smimeRtn,			// from libsecurity_smime
	OSStatus defaultRtn = 0);	// use this if we can't map smimeRtn

#define CFRELEASE(cfr)	if(cfr != NULL) { CFRelease(cfr); }

#include <security_utilities/simulatecrash_assert.h>
#define ASSERT(s) assert(s)

#define CMS_DEBUG 0
#if	CMS_DEBUG
#define CSSM_PERROR(s, r)	cssmPerror(s, r)
#define dprintf(args...)	printf(args)
#else
#define CSSM_PERROR(s, r)
#define dprintf(args...)
#endif

#ifdef __cplusplus
}
#endif

#endif	/* _CMS_UTILS_H_ */

