/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 6, 2021.
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
#include <KerberosHelper/KerberosHelper.h>
#include <CoreFoundation/CoreFoundation.h>
#include <unistd.h>
#include <string.h>
#include <getopt.h>
#include "utils.h"

int main (int argc, char **argv) {
	OSStatus err = 0;
	void *krbHelper = NULL;

	CFStringRef inHostName = NULL, inAdvertisedPrincipal = NULL;
	CFStringRef outRealm = NULL, outServer = NULL, outNoCanon = NULL;
	CFDictionaryRef outDict = NULL;
	const char *inHostNameString = NULL, *inAdvertisedPrincipalString = NULL;

	if (argc > 0) {
		inHostNameString = argv[0];
		inHostName = CFStringCreateWithCString (NULL, inHostNameString, kCFStringEncodingASCII);
	}
	if (argc > 1) {
		inAdvertisedPrincipalString = argv[1];
		inAdvertisedPrincipal = CFStringCreateWithCString (NULL, inAdvertisedPrincipalString, kCFStringEncodingASCII);
	}
	
	err = KRBCreateSession (inHostName, inAdvertisedPrincipal, &krbHelper);
	if (noErr != err) { 
		printf ("ERROR=KRBCreateSession %d (\"%s\",\"%s\" ... )\n", (int)err, inHostNameString, inAdvertisedPrincipalString);
		return 1;
	}

	err = KRBCopyRealm (krbHelper, &outRealm);
	if (noErr != err) { 
		printf ("ERROR=%d from KRBCopyRealm ()\n", (int)err);
		return 2;
	}
	if (outRealm == NULL) {
		printf("ERROR=No realm from KRBCopyRealm\n");
                return 3;
        }

	err = KRBCopyServicePrincipalInfo (krbHelper, CFSTR("host"), &outDict);
	if (noErr != err) { 
		printf ("ERROR=%d from KRBCopyServicePrincipal ()\n", (int)err);
		return 2;
	}
	outServer = CFDictionaryGetValue(outDict, kKRBServicePrincipal);
	if (outServer == NULL) {
		printf("ERROR=No realm from KRBCopyServicePrincipal\n");
		return 3;
	}

	outNoCanon = CFDictionaryGetValue(outDict, kKRBNoCanon);

	char *outRealmString = NULL, *outServerString = NULL;
	__KRBCreateUTF8StringFromCFString(outRealm, &outRealmString);
	__KRBCreateUTF8StringFromCFString(outServer, &outServerString);

	printf ("REALM=%s\n", outRealmString);
	printf ("SERVER=%s\n", outServerString);
	printf ("DNS-CANON=%s\n", outNoCanon ? "NO" : "YES");

	__KRBReleaseUTF8String(outRealmString);
	__KRBReleaseUTF8String(outServerString);

	CFRelease(outRealm);
	CFRelease(outDict);
	CFRelease(inHostName);

	KRBCloseSession(krbHelper);

	sleep(100);

	return err;
}
