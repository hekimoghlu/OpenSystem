/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 7, 2023.
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

typedef const struct __attribute__((objc_bridge(id))) __CCPowerSupply *CCPowerSupplyRef;

/// The standard power supply.
extern const CCPowerSupplyRef kCCPowerStandard;

// Attribute: managed +0 result
__attribute__((cf_returns_not_retained))
CCPowerSupplyRef CCPowerSupplyGetDefault();

typedef const struct __attribute__((objc_bridge(id))) __CCRefrigerator *CCRefrigeratorRef;

// Unaudited function: unmanaged result despite name
CCRefrigeratorRef CCRefrigeratorCreate(CCPowerSupplyRef power);

// Attribute: managed +1 result
__attribute__((cf_returns_retained))
CCRefrigeratorRef CCRefrigeratorSpawn(CCPowerSupplyRef power);

// Parameter: managed +0 by default
void CCRefrigeratorOpen(CCRefrigeratorRef fridge);
void CCRefrigeratorClose(CCRefrigeratorRef fridge);

#pragma clang arc_cf_code_audited begin
// Audited function with Copy convention: managed +1 result
CCRefrigeratorRef CCRefrigeratorCopy(CCRefrigeratorRef fridge);

// Audited function without Copy convention: managed +0 result
CCRefrigeratorRef CCRefrigeratorClone(CCRefrigeratorRef fridge);
#pragma clang arc_cf_code_audited end

// Attribute: managed +1 parameter
void CCRefrigeratorDestroy(__attribute__((cf_consumed)) CCRefrigeratorRef);

@interface CCMagnetismModel
// Unattributed method: unmanaged result
- (CCRefrigeratorRef) refrigerator;
// Attribute: managed +0 result
- (CCRefrigeratorRef) getRefrigerator __attribute__((cf_returns_not_retained));
// Attribute: managed +1 result
- (CCRefrigeratorRef) takeRefrigerator __attribute__((cf_returns_retained));
// Attribute: managed +0 result
- (CCRefrigeratorRef) borrowRefrigerator __attribute__((objc_returns_inner_pointer));

// Parameter: managed +0 by default
- (void) setRefrigerator: (CCRefrigeratorRef) refrigerator;
// Attribute: managed +1 parameter
- (void) giveRefrigerator: (__attribute__((cf_consumed)) CCRefrigeratorRef) refrigerator;

@property CCRefrigeratorRef fridgeProp;
- (CCRefrigeratorRef) fridgeProp __attribute__((cf_returns_not_retained));
@end

typedef double CCFloat;

struct CCImpedance {
  CCFloat real, imag;
};
