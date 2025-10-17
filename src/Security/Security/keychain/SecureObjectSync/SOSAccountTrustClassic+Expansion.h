/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 14, 2022.
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
//  SOSAccountTrustClassic+Expansion_h
//  Security
//
//

#ifndef SOSAccountTrustClassic_Expansion_h
#define SOSAccountTrustClassic_Expansion_h


#import "keychain/SecureObjectSync/SOSAccountTrustClassic.h"

#include <Security/SecureObjectSync/SOSViews.h>
#import "keychain/SecureObjectSync/SOSTransportCircleKVS.h"

@interface SOSAccountTrustClassic (Expansion)

//Expansion Dictionary
//ring handling
-(bool) updateV2Dictionary:(SOSAccount*)account v2:(CFDictionaryRef) newV2Dict;
-(bool) handleUpdateRing:(SOSAccount*)account prospectiveRing:(SOSRingRef)prospectiveRing transport:(SOSKVSCircleStorageTransport*)circleTransport userPublicKey:(SecKeyRef)userPublic writeUpdate:(bool)writeUpdate err:(CFErrorRef *)error;
-(bool) resetRing:(SOSAccount*)account ringName:(CFStringRef) ringName err:(CFErrorRef *)error;
-(bool) resetAccountToEmpty:(SOSAccount*)account transport: (SOSCircleStorageTransport*)circleTransport err:(CFErrorRef*) error;

-(SOSRingRef) copyRing:(CFStringRef) ringName err:(CFErrorRef *)error;
-(CFMutableDictionaryRef) getRings:(CFErrorRef *)error;
-(bool) forEachRing:(RingNameBlock)block;
-(bool) setRing:(SOSRingRef) addRing ringName:(CFStringRef) ringName err:(CFErrorRef*)error;
//generic expansion
-(bool) ensureExpansion:(CFErrorRef *)error;
-(bool) clearValueFromExpansion:(CFStringRef) key err:(CFErrorRef *)error;
-(bool) setValueInExpansion:(CFStringRef) key value:(CFTypeRef) value err:(CFErrorRef *)error;
-(CFTypeRef) getValueFromExpansion:(CFStringRef)key err:(CFErrorRef*)error;
-(void) setRings:(CFMutableDictionaryRef) newrings;
-(bool) valueSetContainsValue:(CFStringRef) key value:(CFTypeRef) value;
-(void) valueUnionWith:(CFStringRef) key valuesToUnion:(CFSetRef) valuesToUnion;
-(void) valueSubtractFrom:(CFStringRef) key valuesToSubtract:(CFSetRef) valuesToSubtract;
-(void) pendEnableViewSet:(CFSetRef) enabledViews;
-(bool) resetAllRings:(SOSAccount*)account err:(CFErrorRef *)error;
-(bool) checkForRings:(CFErrorRef*)error;

@end

#endif /* SOSAccountTrustClassic_Expansion_h */
