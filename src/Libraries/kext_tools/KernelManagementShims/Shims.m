/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 29, 2023.
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
#import <sysexits.h>

#import <pthread/pthread.h>
#import <Foundation/Foundation.h>

#if !(TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR)
#import <KernelManagement/KernelManagement.h>
#import <KernelManagement/KernelManagement_Private.h>

#import "kextcache_main.h"
#import "kext_tools_util.h"
#endif

bool isKernelManagementLinked() {
#if TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR
    /* KM doesn't exist on iPhone */
    return false;
#else
    return NSClassFromString(@"KernelManagementClient") ? true : false;
#endif
}

int KernelManagementLoadKextsWithURLs(CFArrayRef urls)
{
#if TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR
	(void)urls;
	return EX_OSERR;
#else
	int result = EX_OSERR;
	int count = (int)CFArrayGetCount(urls);

	NSMutableArray<NSString *> *paths = nil;
	NSArray<NSURL *> *nsurls = nil;
	NSError *error = nil;

	paths = [NSMutableArray arrayWithCapacity:count];
	if (!paths) {
		goto finish;
	}

	nsurls = (__bridge NSArray<NSURL *> *)urls;
	for (int i = 0; i < count; i++) {
		paths[i] = nsurls[i].path;
	}

	if (![[KernelManagementClient sharedClient] loadExtensionsWithPaths:paths withError:&error]) {
		OSKextLogCFString(/* kext */ NULL,
			kOSKextLogErrorLevel | kOSKextLogIPCFlag,
			CFSTR("Error contacting KernelManagement service: %@"),
			(__bridge CFStringRef)error.localizedDescription);
		goto finish;
	}

	result = EX_OK;
finish:
	return result;
#endif // #if TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR
}
