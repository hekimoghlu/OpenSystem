/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 14, 2023.
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
#include "utils.h"

int main (int argc, const char * argv[]) {
	OSStatus err = 0;
	void *krbHelper = NULL;
	CFStringRef localRealm = NULL, outPrincipal = NULL, outUsername = NULL, inUsername = NULL, hostName = NULL, hostNameDotLocal = NULL;
	char *output, *ptr;
	int testNumber = -1, lineNumber = 0, cases;
	CFDictionaryRef			outDict = NULL;
	CFMutableDictionaryRef	inDict = NULL;
	char myHostname[_POSIX_HOST_NAME_MAX]; // XXX HOST_NAME_MAX
	char *myHostnameLocal;
	
	if (0 != gethostname (myHostname, sizeof(myHostname))) { lineNumber = __LINE__; goto Error; }

	ptr = strchr (myHostname, '.'); 
	if (NULL != ptr) { *ptr = '\0'; }
	
	asprintf (&myHostnameLocal, "%s.local", myHostname);
	
	hostNameDotLocal = CFStringCreateWithCString (NULL, myHostnameLocal, kCFStringEncodingASCII);
	hostName         = CFStringCreateWithCString (NULL, myHostname,      kCFStringEncodingASCII);

	testNumber = 0;
	/*******************************************************************************************/
			
	/* First test the "local" behaviour */
	err = KRBCreateSession (NULL, NULL, &krbHelper);
	if (noErr != err) { lineNumber = __LINE__; goto Error; }

	err = KRBCopyRealm (krbHelper, &localRealm);
	if (noErr != err) { lineNumber = __LINE__; goto Error; }

	__KRBCreateUTF8StringFromCFString (localRealm, &output);
	
	printf ("[%d] Local Realm = %s\n\n", testNumber, output);
	__KRBReleaseUTF8String (output);

	KRBCloseSession (krbHelper);
	testNumber++;

	/*******************************************************************************************/
			
	/* First test the "local" behaviour */
	for (cases = 0; cases < 2; cases++) {

		err = KRBCreateSession (NULL, NULL, &krbHelper);
		if (noErr != err) { lineNumber = __LINE__; goto Error; }

		err = KRBCopyKeychainLookupInfo (krbHelper, inUsername, &outDict);
		if (noErr != err) { lineNumber = __LINE__; goto Error; }

		printf ("[%d] CopyKeychainLookupInfo dictionary...\n", testNumber);
		CFShow (outDict);
		printf ("[%d] \n\n", testNumber);
	
		KRBCloseSession (krbHelper);
		testNumber++;

		inUsername = CFStringCreateWithCString (NULL, getlogin(), kCFStringEncodingASCII);
	}

	/*******************************************************************************************/
	err = KRBCreateSession (hostNameDotLocal, NULL, &krbHelper);
	if (noErr != err) { lineNumber = __LINE__; goto Error; }

	err = KRBCopyServicePrincipal (krbHelper, CFSTR("afpserver"), &outPrincipal);
	if (noErr != err) { lineNumber = __LINE__; goto Error; }
	
	__KRBCreateUTF8StringFromCFString (outPrincipal, &output);
	printf ("[%d] ServicePrincipal = %s\n\n", testNumber, output);
	__KRBReleaseUTF8String (output);

	KRBCloseSession (krbHelper);
	testNumber++;

	/*******************************************************************************************/
	err = KRBCreateSession (CFSTR("17.202.44.91"), CFSTR("afpserver/homedepot.apple.com@OD.APPLE.COM"), &krbHelper);
	if (noErr != err) { lineNumber = __LINE__; goto Error; }
		
	err = KRBCopyServicePrincipal (krbHelper, CFSTR("afpserver"), &outPrincipal);
	if (noErr != err) { lineNumber = __LINE__; goto Error; }
		
	__KRBCreateUTF8StringFromCFString (outPrincipal, &output);
	printf ("[%d] ServicePrincipal = %s\n\n", testNumber, output);
	__KRBReleaseUTF8String (output);
	
	KRBCloseSession (krbHelper);
	testNumber++;

	/*******************************************************************************************/
	err = KRBCreateSession (hostNameDotLocal, NULL, &krbHelper);
	if (noErr != err) { lineNumber = __LINE__; goto Error; }
		
	err = KRBCopyServicePrincipal (krbHelper, CFSTR("afpserver"), &outPrincipal);
	if (noErr != err) { lineNumber = __LINE__; goto Error; }
		
	__KRBCreateUTF8StringFromCFString (outPrincipal, &output);
	printf ("[%d] ServicePrincipal = %s\n\n", testNumber, output);
	__KRBReleaseUTF8String (output);
	
	err = KRBCopyClientPrincipalInfo (krbHelper, inDict, &outDict);
	if (noErr != err) { lineNumber = __LINE__; goto Error; }
		
	outPrincipal = CFDictionaryGetValue (outDict, kKRBClientPrincipalKey);
	
	__KRBCreateUTF8StringFromCFString (outPrincipal, &output);
	printf ("[%d] ClientPrincipal  = %s\n\n", testNumber, output);
	__KRBReleaseUTF8String (output);
	
	KRBCloseSession (krbHelper);
	testNumber++;

	/*******************************************************************************************/
#if 0
	err = KRBCreateSession (CFSTR("statler"), NULL, &krbHelper);
	if (noErr != err) { lineNumber = __LINE__; goto Error; }
		
	err = KRBCopyServicePrincipal (krbHelper, CFSTR("cifs"), &outPrincipal);
	if (noErr != err) { lineNumber = __LINE__; goto Error; }
		
	__KRBCreateUTF8StringFromCFString (outPrincipal, &output);
	printf ("[%d] ServicePrincipal = %s\n\n", testNumber, output);
	__KRBReleaseUTF8String (output);

	err = KRBCopyClientPrincipalInfo (krbHelper, inDict, &outDict);
	if (noErr != err) { lineNumber = __LINE__; goto Error; }
		
	outPrincipal = CFDictionaryGetValue (outDict, kKRBClientPrincipalKey);
	
	__KRBCreateUTF8StringFromCFString (outPrincipal, &output);
	printf ("[%d] ClientPrincipal  = %s\n\n", testNumber, output);
	__KRBReleaseUTF8String (output);
	KRBCloseSession (krbHelper);
	testNumber++;
#endif	
	/*******************************************************************************************/
	err = KRBCreateSession (hostNameDotLocal, CFSTR("afpserver/kerberos@OD.APPLE.COM"), &krbHelper);
	if (noErr != err) { lineNumber = __LINE__; goto Error; }

	err = KRBCopyServicePrincipal (krbHelper, NULL, &outPrincipal);
	if (noErr != err) { lineNumber = __LINE__; goto Error; }

	__KRBCreateUTF8StringFromCFString (outPrincipal, &output);
	printf ("[%d] ServicePrincipal = %s\n\n", testNumber, output);
	__KRBReleaseUTF8String (output);

	KRBCloseSession (krbHelper);
	testNumber++;

	/*******************************************************************************************/
	err = KRBCreateSession (hostName, CFSTR("afpserver/kerberos@OD.APPLE.COM"), &krbHelper);
	if (noErr != err) { lineNumber = __LINE__; goto Error; }

	err = KRBCopyServicePrincipal (krbHelper, NULL, &outPrincipal);
	if (noErr != err) { lineNumber = __LINE__; goto Error; }

	__KRBCreateUTF8StringFromCFString (outPrincipal, &output);
	printf ("[%d] ServicePrincipal = %s\n\n", testNumber, output);
	__KRBReleaseUTF8String (output);

	KRBCloseSession (krbHelper);
	testNumber++;

	/*******************************************************************************************/
	err = KRBCreateSession (hostNameDotLocal, NULL, &krbHelper);
	if (noErr != err) { lineNumber = __LINE__; goto Error; }

	err = KRBCopyServicePrincipal (krbHelper, CFSTR("afpserver"), &outPrincipal);
	if (noErr != err) { lineNumber = __LINE__; goto Error; }
	
	__KRBCreateUTF8StringFromCFString (outPrincipal, &output);
	printf ("[%d] ServicePrincipal = %s\n", testNumber, output);
	__KRBReleaseUTF8String (output);

	err = KRBCopyClientPrincipalInfo (krbHelper, inDict, &outDict);
	if (noErr != err) { lineNumber = __LINE__; goto Error; }
	
	outPrincipal = CFDictionaryGetValue (outDict, kKRBClientPrincipalKey);

	__KRBCreateUTF8StringFromCFString (outPrincipal, &output);
	printf ("[%d] ClientPrincipal  = %s\n\n", testNumber, output);
	__KRBReleaseUTF8String (output);

	KRBCloseSession (krbHelper);
	testNumber++;
	
	/*******************************************************************************************/
	err = KRBCreateSession (hostNameDotLocal, NULL, &krbHelper);
	if (noErr != err) { lineNumber = __LINE__; goto Error; }

	err = KRBCopyServicePrincipal (krbHelper, CFSTR("afpserver"), &outPrincipal);
	if (noErr != err) { lineNumber = __LINE__; goto Error; }
	
	__KRBCreateUTF8StringFromCFString (outPrincipal, &output);
	printf ("[%d] ServicePrincipal = %s\n", testNumber, output);
	__KRBReleaseUTF8String (output);

	inDict = CFDictionaryCreateMutable (kCFAllocatorDefault, 4, &kCFCopyStringDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);

	CFDictionarySetValue (inDict, kKRBUsernameKey,		 CFSTR("remoteUsername"));
	CFDictionarySetValue (inDict, kKRBClientPasswordKey, CFSTR("remotePassword"));
	CFDictionarySetValue (inDict, kKRBAllowKerberosUI,	 kKRBOptionNoUI);

	err = KRBCopyClientPrincipalInfo (krbHelper, inDict, &outDict);
	if (noErr != err) { lineNumber = __LINE__; goto Error; }
	
	outPrincipal = CFDictionaryGetValue (outDict, kKRBClientPrincipalKey);

	__KRBCreateUTF8StringFromCFString (outPrincipal, &output);
	printf ("[%d] ClientPrincipal  = %s\n\n", testNumber, output);
	__KRBReleaseUTF8String (output);

	KRBCloseSession (krbHelper);
	testNumber++;
	
	/*******************************************************************************************/
        for (cases = 0; cases < 2; cases++) {
                err = KRBCreateSession (hostNameDotLocal, NULL, &krbHelper);
                if (noErr != err) { lineNumber = __LINE__; goto Error; }

                err = KRBCopyServicePrincipal (krbHelper, CFSTR("afpserver"), &outPrincipal);
                if (noErr != err) { lineNumber = __LINE__; goto Error; }

                __KRBCreateUTF8StringFromCFString (outPrincipal, &output);
                printf ("[%d] ServicePrincipal = %s\n", testNumber, output);
                __KRBReleaseUTF8String (output);

                inDict = CFDictionaryCreateMutable (kCFAllocatorDefault, 4, &kCFCopyStringDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);

                CFDictionarySetValue (inDict, kKRBAllowKerberosUI,       kKRBOptionNoUI);

                err = KRBCopyClientPrincipalInfo (krbHelper, inDict, &outDict);
		CFRelease(inDict);
                if (noErr != err) { lineNumber = __LINE__; goto Error; }

                outPrincipal = CFDictionaryGetValue (outDict, kKRBClientPrincipalKey);
                outUsername  = CFDictionaryGetValue (outDict, kKRBUsernameKey);

                CFDictionarySetValue ((CFMutableDictionaryRef)outDict, kKRBClientPasswordKey, outUsername);

                __KRBCreateUTF8StringFromCFString (outPrincipal, &output);
                printf ("[%d] ClientPrincipal  = %s\n\n", testNumber, output);
                __KRBReleaseUTF8String (output);

                err = KRBTestForExistingTicket (krbHelper, outDict);
                if (noErr == err) {
                        printf("[%d] Ticket was already available\n", testNumber);
                } else {
			printf("[%d] Ticket was not available (err=%d), trying to obtain one\n", testNumber, (int)err);
                        err = KRBAcquireTicket(krbHelper, outDict);
                        if (noErr != err) { lineNumber = __LINE__; goto Error; }
                }

                KRBCloseSession (krbHelper);
                testNumber++;
        }

Error:
	{
		pid_t	pid = getpid();
		char	*command;
		
		asprintf (&command, "leaks %d", pid);
		system (command);
	}
	if (err) {
		fprintf (stdout, "Error (%d) in test %d at line %d\n", (int)err, testNumber, lineNumber);
		return 1;
	}
	return 0;
}
