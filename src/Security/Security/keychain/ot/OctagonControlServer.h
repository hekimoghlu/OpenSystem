/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 4, 2022.
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
#define kSecuritydOctagonServiceName "com.apple.security.octagon"

__BEGIN_DECLS

void OctagonControlServerInitialize(void);


#if __OBJC__
#import "keychain/ot/OTControlProtocol.h"
NS_ASSUME_NONNULL_BEGIN

@protocol OctagonEntitlementBearerProtocol
- (nullable id)valueForEntitlement:(NSString *)entitlement;
@end

@interface NSXPCConnection (OctagonEntitlement) <OctagonEntitlementBearerProtocol>
@end

#if OCTAGON
@class OTManager;
@interface OctagonXPCEntitlementChecker : NSProxy
+ (id<OTControlProtocol>)createWithManager:(OTManager*)manager
                         entitlementBearer:(id<OctagonEntitlementBearerProtocol>)bearer;
@end
#endif // OCTAGON

NS_ASSUME_NONNULL_END
#endif // __OBJC__

__END_DECLS
