/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 2, 2024.
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
 * main.c
 * - test harness to test IPMonitorControl client and server
 */

/*
 * Modification History
 *
 * December 16, 2013	Dieter Siegmund (dieter@apple.com)
 * - initial revision
 */

#include <stdlib.h>
#include <CoreFoundation/CoreFoundation.h>
#include <SystemConfiguration/SCPrivate.h>

#include "IPMonitorControl.h"
#include "IPMonitorControlServer.h"
#include "symbol_scope.h"

STATIC void
AssertionsChanged(void * info)
{
    CFDictionaryRef 	assertions = NULL;
    CFArrayRef		changes;

    changes = IPMonitorControlServerCopyInterfaceRankInformation(&assertions);
    SCPrint(TRUE, stdout, CFSTR("Changed interfaces %@\n"), changes);
    if (assertions == NULL) {
	SCPrint(TRUE, stdout, CFSTR("No assertions\n"));
    }
    else {
	SCPrint(TRUE, stdout, CFSTR("Assertions = %@\n"), assertions);
	CFRelease(assertions);
    }
    if (changes != NULL) {
	CFRelease(changes);
    }
    return;
}

int
main(int argc, char * argv[])
{
    if (argc >= 2) {
	int				ch;
	IPMonitorControlRef		control;
	SCNetworkServicePrimaryRank 	rank;
	Boolean				rank_set = FALSE;
	Boolean				wait = FALSE;

	rank = kSCNetworkServicePrimaryRankDefault;
	control = IPMonitorControlCreate();
	if (control == NULL) {
	    fprintf(stderr, "failed to allocate IPMonitorControl\n");
	    exit(1);
	}

	while ((ch = getopt(argc, argv, "i:r:w")) != EOF) {
	    CFStringRef			ifname;
	    SCNetworkServicePrimaryRank	existing_rank;

	    switch ((char)ch) {
	    case 'i':
		ifname = CFStringCreateWithCString(NULL, optarg,
						   kCFStringEncodingUTF8);
		existing_rank = IPMonitorControlGetInterfacePrimaryRank(control,
									ifname);
		printf("%s rank was %u\n", optarg, existing_rank);
		if (IPMonitorControlSetInterfacePrimaryRank(control,
							    ifname,
							    rank)) {
		    printf("%s rank set to %u\n", optarg, rank);
		    rank_set = TRUE;
		}
		else {
		    fprintf(stderr, "failed to set rank\n");
		}
		CFRelease(ifname);
		break;
	    case 'r':
		rank = strtoul(optarg, NULL, 0);
		break;
	    case 'w':
		wait = TRUE;
		break;
	    default:
		fprintf(stderr, "unexpected option '%c'\n", (char)ch);
		exit(1);
		break;
	    }
	}
	argc -= optind;
	argv += optind;
	if (argc > 0) {
	    fprintf(stderr, "ignoring additional parameters\n");
	}
	if (!rank_set) {
	    exit(1);
	}
	if (wait) {
	    CFRunLoopRun();
	}
    }
    else {
	CFRunLoopSourceContext 	context;
	CFRunLoopSourceRef	rls;
	STATIC Boolean		verbose = TRUE;

	memset(&context, 0, sizeof(context));
	context.info = (void *)NULL;
	context.perform = AssertionsChanged;
	rls = CFRunLoopSourceCreate(NULL, 0, &context);
	CFRunLoopAddSource(CFRunLoopGetCurrent(), rls,
			   kCFRunLoopDefaultMode);
	if (!IPMonitorControlServerStart(CFRunLoopGetCurrent(), rls, &verbose)) {
	    fprintf(stderr, "failed to create connection\n");
	    exit(1);
	}
	CFRunLoopRun();
    }
    exit(0);
    return (0);
}

