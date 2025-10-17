/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 5, 2025.
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
 *  BLGetOSVersion.c
 *  bless
 *
 */

#include <sys/param.h>
#include "bless.h"
#include "bless_private.h"
#include "sharedUtilities.h"


int BLGetOSVersion(BLContextPtr context, const char *mount, BLVersionRec *version)
{
	int				err = 0;
	char			fullpath[MAXPATHLEN];
	CFURLRef		plistURL = NULL;
	CFDataRef		versData = NULL;
	CFDictionaryRef	versDict = NULL;
	CFStringRef		versString;
	CFArrayRef		versArray = NULL;
    CFErrorRef      cf_error = NULL;
	
	snprintf(fullpath, sizeof fullpath, "%s/%s", mount, kBL_PATH_SYSTEM_VERSION_PLIST);
	plistURL = CFURLCreateFromFileSystemRepresentation(kCFAllocatorDefault, (const UInt8 *)fullpath, strlen(fullpath), 0);
	if (!plistURL) {
		contextprintf(context, kBLLogLevelError, "Can't get URL for \"%s\"\n", fullpath);
		err = 1;
		goto exit;
	}

    versData = CreateDataFromFileURL(kCFAllocatorDefault, plistURL, &cf_error);
	if (cf_error) {
		contextprintf(context, kBLLogLevelError, "Can't load \"%s\"\n", fullpath);
		err = 2;
		goto exit;
	}
	versDict = CFPropertyListCreateWithData(kCFAllocatorDefault, versData, 0, NULL, NULL);
	if (!versDict) {
		contextprintf(context, kBLLogLevelError, "Could not recognize contents of \"%s\" as a property list\n", fullpath);
		err = 3;
		goto exit;
	}
	versString = CFDictionaryGetValue(versDict, CFSTR("ProductVersion"));
	if (!versString) {
		contextprintf(context, kBLLogLevelError, "Version plist \"%s\" missing ProductVersion item\n", fullpath);
		err = 4;
		goto exit;
	}
	versArray = CFStringCreateArrayBySeparatingStrings(kCFAllocatorDefault, versString, CFSTR("."));
	if (!versArray || CFArrayGetCount(versArray) < 2) {
		contextprintf(context, kBLLogLevelError, "Badly formed version string in plist \"%s\"\n", fullpath);
		err = 5;
		goto exit;
	}
	if (version) {
		version->major = CFStringGetIntValue(CFArrayGetValueAtIndex(versArray, 0));
		version->minor = CFStringGetIntValue(CFArrayGetValueAtIndex(versArray, 1));
		version->patch = (CFArrayGetCount(versArray) >= 3) ? CFStringGetIntValue(CFArrayGetValueAtIndex(versArray, 2)) : 0;
	}
	
exit:
	if (plistURL) CFRelease(plistURL);
	if (versData) CFRelease(versData);
	if (versDict) CFRelease(versDict);
	if (versArray) CFRelease(versArray);
	return err;
}
