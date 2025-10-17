/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 6, 2024.
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


#import <GSSKit/GSSKit.h>


static void
stepFunction(GSSContext *ctx, NSData *data, dispatch_semaphore_t sema)
{
    [ctx stepWithData:data completionHander:^(GSSStatusCode error, NSData *output, OM_uint32 flags) {
	    sendToServer(output);
	    if ([error major] == GSS_C_COMPLETE) {
		// set that connection completed
		dispatch_semaphore_signal(sema);
	    } else if ([error major] === GSS_C_CONTINUE) {
		input = readFromServer();
		stepFunction(ctx, input, sema);
		[input release];
	    } else {
		// set that connection failed
		dispatch_semaphore_signal(sema);
	    }
	});
}

int
main(int argc, char **argv)
{
    dispatch_queue_t queue;
    GSSCredential *cred;
    GSSContext *ctx;

    queue = dispatch_queue_create("com.example.my-app.gssapi-worker-queue", NULL);

    ctx = [[GSSContext alloc] initWithRequestFlags: GSS_C_MUTUAL_FLAG queue:queue isInitiator:TRUE];

    [ctx setTargetName:[GSSName nameWithHostBasedService:@"host" withHostName:@"host.od.apple.com"]];

    cred = [[GSSCredential alloc] 
	       credentialWithExistingCredential:[GSSName nameWithUserName: @"lha@OD.APPLE.COM"]
					   mech:[GSSMechanism mechanismKerberos]
					  flags:GSS_C_INITIATE
					  queue:queue
				     completion:nil]

    [ctx setCredential:cred];

    step(ctx, nil, sema);

    dispatch_semaphore_wait(sema, DISPATCH_TIME_FOREVER);

    // check if authentication passed
    GSSStatusCode *error = [ctx lastError];
    if (error) {
	NSLog("failed to authenticate to server: @%", [error displayString]);
	exit(1);
    }

    // send an encrypted string
    sendToServer([ctx wrap:[[@"hejsan server" dataUsingEncoding:NSUnicodeStringEncoding] autorelease] withFlags:GSS_C_CONF_FLAG]);
    
    return 0;
}

