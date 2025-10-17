/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 19, 2023.
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

@import ObjectiveC;

typedef const void *CFTypeRef __attribute__((objc_bridge(id)));

typedef const struct __attribute__((objc_bridge(id))) CCPowerSupply *CCPowerSupplyRef;
typedef const struct __attribute__((objc_bridge(id))) __CCRefrigerator *CCRefrigeratorRef;
typedef struct __CCRefrigerator *CCMutableRefrigeratorRef;

#pragma clang arc_cf_code_audited begin
_Nonnull CCPowerSupplyRef CCPowerSupplyCreate(double watts)
  __attribute__((language_name("CCPowerSupply.init(watts:)")));

_Nonnull CCRefrigeratorRef CCRefrigeratorCreate(CCPowerSupplyRef _Nonnull power)
  __attribute__((language_name("CCRefrigerator.init(powerSupply:)")));

void CCRefrigeratorOpen(_Null_unspecified CCRefrigeratorRef fridge)
  __attribute__((language_name("CCRefrigerator.open(self:)")));

_Nonnull CCMutableRefrigeratorRef CCRefrigeratorCreateMutable(_Nonnull CCPowerSupplyRef power)
  __attribute__((language_name("CCMutableRefrigerator.init(powerSupply:)")));

_Nonnull CCPowerSupplyRef CCRefrigeratorGetPowerSupply(_Null_unspecified CCRefrigeratorRef fridge)
  __attribute__((language_name("getter:CCRefrigerator.powerSupply(self:)")));

void CCRefrigeratorSetPowerSupply(_Null_unspecified CCRefrigeratorRef fridge,
                                  CCPowerSupplyRef _Nonnull powerSupply)
  __attribute__((language_name("setter:CCRefrigerator.powerSupply(self:_:)")));

extern const _Null_unspecified CCPowerSupplyRef kCCPowerSupplySemiModular
  __attribute__((language_name("CCPowerSupplyRef.semiModular")));

_Nonnull CCPowerSupplyRef CCPowerSupplyCreateDangerous(void)
  __attribute__((language_name("CCPowerSupply.init(dangerous:)")));
#pragma clang arc_cf_code_audited end

extern const double kCCPowerSupplyDefaultPower
  __attribute__((language_name("CCPowerSupply.defaultPower")));

extern const _Nonnull CCPowerSupplyRef kCCPowerSupplyAC
  __attribute__((language_name("CCPowerSupply.AC")));

extern const _Nullable CCPowerSupplyRef kCCPowerSupplyDC
  __attribute__((language_name("CCPowerSupply.DC")));
