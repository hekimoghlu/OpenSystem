/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 3, 2023.
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
#import <Foundation/NSXPCConnection_Private.h>

#import "IPConfigurationLog.h"
#import "IPConfigurationPrivate.h"
#import "IPHPvDInfoRequest.h"

static NSString * const kIPHPvDInfoRequestClientEntitlement = @"com.apple.private.IPConfigurationHelper.PvD";

@interface IPConfigurationHelperDelegate : NSObject<NSXPCListenerDelegate>
@end

@implementation IPConfigurationHelperDelegate

- (instancetype)init
{
	self = [super init];
	if (self) {
		_IPConfigurationInitLog(kIPConfigurationLogCategoryHelper);
	}
	return self;
}

- (BOOL)listener:(NSXPCListener *)listener shouldAcceptNewConnection:(NSXPCConnection *)newConnection {
	id clientEntitlement = nil;
	BOOL acceptNewConnection = NO;

	clientEntitlement = [newConnection valueForEntitlement:kIPHPvDInfoRequestClientEntitlement];
	if (clientEntitlement == nil || ![clientEntitlement isKindOfClass:[NSNumber class]]) {
		goto done;
	}
	acceptNewConnection = ((NSNumber *)clientEntitlement).boolValue;
	if (!acceptNewConnection) {
		IPConfigLog(LOG_NOTICE, "rejecting new connection due to missing entitlement");
		goto done;
	}
	newConnection.exportedObject = [IPHPvDInfoRequestServer new];
	newConnection.exportedInterface = [NSXPCInterface interfaceWithProtocol:
					   @protocol(IPHPvDInfoRequestProtocol)];
	[newConnection resume];

done:
	return acceptNewConnection;
}

@end

int main(int argc, const char * argv[])
{
	IPConfigurationHelperDelegate *delegate = nil;
	NSXPCListener *listener = nil;

	/*
	 * An app-scoped service inherits the app's user, which is root for configd.
	 * This hops out of root (0) and instead runs this XPCService as user nobody (-2).
	 */
	if (geteuid() == 0) {
		if (seteuid(-2) != 0) {
			IPConfigLog(LOG_ERR, "couldn't deescalate user before launching");
			goto done;
		}
	}
	/* XPC server start */
	delegate = [IPConfigurationHelperDelegate new];
	listener = [NSXPCListener serviceListener];
	listener.delegate = delegate;
	[listener activate];

done:
	return 0;
}
