/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 26, 2025.
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
#include <sys/param.h>
#include <sys/errno.h>
#include <sys/stat.h>
#include <err.h>
#include <stdio.h>
#include <unistd.h>
#include <strings.h>
#include <stdlib.h>
#include <sysexits.h>

#include <smbclient/smbclient.h>
#include <smbclient/ntstatus.h>

#include "common.h"
#include "netshareenum.h"

char *CStringCreateWithCFString(CFStringRef inStr)
{
	CFIndex maxLen;
	char *str;
	
	if (inStr == NULL) {
		return NULL;
	}
	maxLen = CFStringGetMaximumSizeForEncoding(CFStringGetLength(inStr), 
											   kCFStringEncodingUTF8) + 1;
	str = malloc(maxLen);
	if (!str) {
		return NULL;
	}
	CFStringGetCString(inStr, str, maxLen, kCFStringEncodingUTF8);
	return str;
}

/*
 * Given a share dictionary create an array that contains the share entries in
 * the dictionary.
 */
CFArrayRef createShareArrayFromShareDictionary(CFDictionaryRef shareDict)
{
	CFIndex count = 0;
	CFArrayRef keyArray = NULL;
    
    if (!shareDict)
        return NULL;
    count = CFDictionaryGetCount(shareDict);
    
    void *shareKeys = CFAllocatorAllocate(kCFAllocatorDefault, count * sizeof(CFStringRef), 0);
	if (shareKeys) {
		CFDictionaryGetKeysAndValues(shareDict, (const void **)shareKeys, NULL);
		keyArray = CFArrayCreate(kCFAllocatorDefault, (const void **)shareKeys,
                                 count, &kCFTypeArrayCallBacks);
		CFAllocatorDeallocate(kCFAllocatorDefault, shareKeys);
	}
    
	return keyArray;
}

int
cmd_view(int argc, char *argv[])
{
	const char *url = NULL;
	int			opt;
	SMBHANDLE	serverConnection = NULL;
	uint64_t	options = 0;
	NTSTATUS	status;
	int			error;
	CFDictionaryRef shareDict= NULL;
	
	while ((opt = getopt(argc, argv, "ANGgaf")) != EOF) {
		switch(opt){
			case 'A':
				options |= kSMBOptionSessionOnly;
				break;
			case 'N':
				options |= kSMBOptionNoPrompt;
				break;
			case 'G':
				options |= kSMBOptionAllowGuestAuth;
				break;
			case 'g':
				if (options & kSMBOptionOnlyAuthMask)
					view_usage();
				options |= kSMBOptionUseGuestOnlyAuth;
				options |= kSMBOptionNoPrompt;
				break;
			case 'a':
				if (options & kSMBOptionOnlyAuthMask)
					view_usage();
				options |= kSMBOptionUseAnonymousOnlyAuth;
				options |= kSMBOptionNoPrompt;
				break;
			case 'f':
				options |= kSMBOptionForceNewSession;
				break;
			default:
				view_usage();
				/*NOTREACHED*/
		}
	}
	
	if (optind >= argc)
		view_usage();
	url = argv[optind];
	argc -= optind;
	/* One more check to make sure we have the correct number of arguments */
	if (argc != 1)
		view_usage();
	
	status = SMBOpenServerEx(url, &serverConnection, options);
	/* 
	 * SMBOpenServerEx now sets errno, so err will work correctly. We change 
	 * the string based on the NTSTATUS Error.
	 */
	if (!NT_SUCCESS(status)) {
		/* This routine will exit the program */
		ntstatus_to_err(status);
	}
	if (options  & kSMBOptionSessionOnly) {
		fprintf(stdout, "Authenticate successfully with %s\n", url);
		goto done;
	}
	fprintf(stdout, "%-48s%-8s%s\n", "Share", "Type", "Comments");
	fprintf(stdout, "-------------------------------\n");

	error = smb_netshareenum(serverConnection, &shareDict, FALSE);
	if (error) {
		errno = error;
		SMBReleaseServer(serverConnection);
		err(EX_IOERR, "unable to list resources");
	} else {
		CFArrayRef shareArray = createShareArrayFromShareDictionary(shareDict);
		CFStringRef shareStr, shareTypeStr, commentStr;
		CFDictionaryRef theDict;
		CFIndex ii;
		char *share, *sharetype, *comments;
						
		for (ii=0; shareArray && (ii < CFArrayGetCount(shareArray)); ii++) {
			shareStr = CFArrayGetValueAtIndex(shareArray, ii);
			/* Should never happen, but just to be safe */
			if (shareStr == NULL) {
				continue;
			}
			theDict = CFDictionaryGetValue(shareDict, shareStr);
			/* Should never happen, but just to be safe */
			if (theDict == NULL) {
				continue;
			}
			shareTypeStr = CFDictionaryGetValue(theDict, kNetShareTypeStrKey);
			commentStr = CFDictionaryGetValue(theDict, kNetCommentStrKey);
			
			share = CStringCreateWithCFString(shareStr);
			sharetype = CStringCreateWithCFString(shareTypeStr);
			comments = CStringCreateWithCFString(commentStr);
			fprintf(stdout, "%-48s%-8s%s\n", share ? share : "",  
					sharetype ? sharetype : "", comments ? comments : "");
			free(share);
			free(sharetype);
			free(comments);
		}
		if (shareArray) {
			fprintf(stdout, "\n%ld shares listed\n", CFArrayGetCount(shareArray));
			CFRelease(shareArray);
		} else {
			fprintf(stdout, "\n0 shares listed\n");
		}
		if (shareDict) {
			CFRelease(shareDict);
		}
	}
done:
	SMBReleaseServer(serverConnection);
	return 0;
}


void
view_usage(void)
{
	fprintf(stderr, "usage: smbutil view [connection options] //"
		"[domain;][user[:password]@]"
	"server\n");
	
	fprintf(stderr, "where options are:\n"
					"    -A    authorize only\n"
					"    -N    don't prompt for a password\n"
					"    -G    allow guest access\n"
					"    -g    authorize with guest only\n"
					"    -a    authorize with anonymous only\n"
					"    -f    don't share session\n");
	exit(1);
}