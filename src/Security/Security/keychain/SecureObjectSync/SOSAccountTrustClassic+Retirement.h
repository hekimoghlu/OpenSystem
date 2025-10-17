/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 5, 2023.
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
//  SOSAccountTrustClassic+Retirement.h
//  Security
//

#ifndef SOSAccountTrustClassic_Retirement_h
#define SOSAccountTrustClassic_Retirement_h

#import "keychain/SecureObjectSync/SOSTransportCircleKVS.h"
#import "keychain/SecureObjectSync/SOSAccountTrustClassic.h"

@interface SOSAccountTrustClassic (Retirement)
//Retirement
-(bool) cleanupAfterPeer:(SOSMessageKVS*)kvsTransport circleTransport:(SOSCircleStorageTransport*)circleTransport seconds:(size_t) seconds circle:(SOSCircleRef) circle cleanupPeer:(SOSPeerInfoRef) cleanupPeer err:(CFErrorRef*) error;
-(bool) cleanupRetirementTickets:(SOSAccount*)account circle:(SOSCircleRef)circle time:(size_t) seconds err:(CFErrorRef*) error;
@end
#endif /* SOSAccountTrustClassic_Retirement_h */
