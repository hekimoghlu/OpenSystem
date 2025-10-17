/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 30, 2025.
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
#import <Foundation/Foundation.h>


int main(int argc, const char * argv[]) {
    @autoreleasepool {

	if (argc != 2)
	{
	    printf("gsstoolwrapper [audit token file name] \n");
	    printf("This program is a quick way to get a \"real\" audit token to pass to gsstool for testing. The pid from it has to be alive for it to be valid for gss use. The intended use is to run the app and then bacground it.  It will stay alive so that gsstool can use the audit token to confirm that delegation is working correctly based on the current configuration.  If the wrapper tool is assigned per app vpn, then gss tool should be able to confirm delegation is working correctly.\n");
	}
	
	NSString *tempFilePath = [NSString stringWithCString:argv[1] encoding:NSUTF8StringEncoding];
	
	audit_token_t self_token;
	mach_msg_type_number_t token_size = TASK_AUDIT_TOKEN_COUNT;
	kern_return_t kr = task_info(mach_task_self(), TASK_AUDIT_TOKEN,
	(integer_t *)&self_token, &token_size);
	if (kr != KERN_SUCCESS) {
	    printf("Failed to get own token");
	    return 1;
	}
	
	NSData *tokenData = [NSData dataWithBytesNoCopy:&self_token length:sizeof(audit_token_t)];
	
	if (![tokenData writeToFile:tempFilePath atomically:YES]) {
	    printf("Failed to write out token");
	    return 1;
	}

	[[NSRunLoop currentRunLoop] run];

    }
    return 0;
}
