/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 28, 2023.
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

//
//  SOSAccountTrustClassic+Identity.h
//  Security
//

#ifndef SOSAccountTrustClassic_Identity_h
#define SOSAccountTrustClassic_Identity_h

#import "keychain/SecureObjectSync/SOSAccountTrustClassic.h"

@class SOSAccountTrustClassic;

@interface SOSAccountTrustClassic (Identity)
//FullPeerInfo
-(bool) updateFullPeerInfo:(SOSAccount*)account minimum:(CFSetRef)minimumViews excluded:(CFSetRef)excludedViews;
-(SOSFullPeerInfoRef) getMyFullPeerInfo;
-(bool) fullPeerInfoVerify:(SecKeyRef) privKey err:(CFErrorRef *)error;
-(bool) hasFullPeerInfo:(CFErrorRef*) error;
-(SOSFullPeerInfoRef) CopyAccountIdentityPeerInfo CF_RETURNS_RETAINED;
-(bool) ensureFullPeerAvailable:(SOSAccount*)account err:(CFErrorRef *) error;
-(bool) isMyPeerActive:(CFErrorRef*) error;
-(void) purgeIdentity;

- (void)ensureOctagonPeerKeys:(SOSKVSCircleStorageTransport*)circleTransport;

@end


#endif /* SOSAccountTrustClassic_Identity_h */
