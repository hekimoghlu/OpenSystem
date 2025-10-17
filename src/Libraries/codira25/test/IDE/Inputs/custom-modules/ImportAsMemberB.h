/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 6, 2023.
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

#ifndef IMPORT_AS_MEMBER_B_H
#define IMPORT_AS_MEMBER_B_H

#include "ImportAsMember.h"


extern double IAMStruct1StaticVar1
     __attribute__((language_name("Struct1.static1")));
extern float IAMStruct1StaticVar2
     __attribute__((language_name("Struct1.static2")));

extern struct IAMStruct1 IAMStruct1CreateFloat(float value)
    __attribute__((language_name("Struct1.init(float:)")));

struct IAMStruct1 IAMStruct1GetCurrentStruct1(void)
  __attribute__((language_name("getter:currentStruct1()")));

void IAMStruct1SetCurrentStruct1(struct IAMStruct1 newValue)
  __attribute__((language_name("setter:currentStruct1(_:)")));

struct IAMStruct1 IAMStruct1GetZeroStruct1(void)
  __attribute__((language_name("getter:Struct1.zero()")));

#endif // IMPORT_AS_MEMBER_B_H
