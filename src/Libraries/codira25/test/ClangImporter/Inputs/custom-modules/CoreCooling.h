/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 22, 2025.
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

typedef const void *CFTypeRef __attribute__((objc_bridge(id)));
CFTypeRef CFBottom();

typedef const struct __attribute__((objc_bridge(id))) __CCPowerSupply *CCPowerSupplyRef;
typedef const struct __attribute__((objc_bridge(id))) __CCItem *CCItemRef;

/// The standard power supply.
extern const CCPowerSupplyRef kCCPowerStandard;

typedef const struct __attribute__((objc_bridge(id))) __CCRefrigerator *CCRefrigeratorRef;
CCRefrigeratorRef CCRefrigeratorCreate(CCPowerSupplyRef power);

void CCRefrigeratorOpen(CCRefrigeratorRef fridge);
CCItemRef CCRefrigeratorGet(CCRefrigeratorRef fridge, unsigned index);
void CCRefrigeratorClose(CCRefrigeratorRef fridge);

typedef struct __CCRefrigerator *CCMutableRefrigeratorRef;
CCMutableRefrigeratorRef CCRefrigeratorCreateMutable(CCPowerSupplyRef power);

void CCRefrigeratorInsert(CCMutableRefrigeratorRef fridge, CCItemRef ref);

@interface Kitchen
@property CCRefrigeratorRef fridge;
-(void)replacePowerSupply:(CCPowerSupplyRef)powerSupply;
@end

@interface Duct
@end

@interface MutableDuct : Duct
@end

typedef const struct __attribute__((objc_bridge(Duct))) __CCDuct *CCDuctRef;
typedef struct __attribute__((objc_bridge_mutable(MutableDuct))) __CCDuct *CCMutableDuctRef;

typedef CCRefrigeratorRef CCFridgeRef;

typedef const void *CCOpaqueTypeRef __attribute__((objc_bridge(id)));
CCOpaqueTypeRef CCRetain(CCOpaqueTypeRef typeRef);
void CCRelease(CCOpaqueTypeRef typeRef);
CCOpaqueTypeRef CCMungeAndRetain(CCOpaqueTypeRef typeRef) __attribute__((language_name("CCMungeAndRetain(_:)")));

// Nullability
void CCRefrigeratorOpenDoSomething(_Nonnull CCRefrigeratorRef fridge);
void CCRefrigeratorOpenMaybeDoSomething(_Nullable CCRefrigeratorRef fridge);

// Out parameters
void CCRefrigeratorCreateIndirect(CCRefrigeratorRef *_Nullable
                                  __attribute__((cf_returns_retained))
                                  outFridge);
// Note that the fridge parameter is incorrectly annotated.
void CCRefrigeratorGetPowerSupplyIndirect(
    CCRefrigeratorRef __attribute__((cf_returns_not_retained)) fridge,
    CCPowerSupplyRef *_Nonnull __attribute__((cf_returns_not_retained))
    outPower);
void CCRefrigeratorGetItemUnaudited(CCRefrigeratorRef fridge, unsigned index, CCItemRef *outItem);

typedef void *CFNonConstVoidRef __attribute__((objc_bridge(id)));
CFNonConstVoidRef CFNonConstBottom();

typedef struct IceCube {
    float width;
    float height;
    float depth;
} IceCube;

typedef IceCube IceCube;
typedef IceCube BlockOfIce;
