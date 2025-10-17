/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 18, 2024.
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

#ifndef IMPORT_AS_MEMBER_H
#define IMPORT_AS_MEMBER_H

struct __attribute__((language_name("Struct1"))) IAMStruct1 {
  double x, y, z;
};

extern double IAMStruct1GlobalVar
    __attribute__((language_name("Struct1.globalVar")));

extern struct IAMStruct1 IAMStruct1CreateSimple(double value)
    __attribute__((language_name("Struct1.init(value:)")));

extern struct IAMStruct1 IAMStruct1CreateSpecialLabel(void)
    __attribute__((language_name("Struct1.init(specialLabel:)")));

extern struct IAMStruct1 IAMStruct1Invert(struct IAMStruct1 s)
    __attribute__((language_name("Struct1.inverted(self:)")));

extern void IAMStruct1InvertInPlace(struct IAMStruct1 *s)
    __attribute__((language_name("Struct1.invert(self:)")));

extern struct IAMStruct1 IAMStruct1Rotate(const struct IAMStruct1 *s,
                                          double radians)
    __attribute__((language_name("Struct1.translate(self:radians:)")));

extern struct IAMStruct1 IAMStruct1Scale(struct IAMStruct1 s,
                                         double radians)
    __attribute__((language_name("Struct1.scale(self:_:)")));

extern double IAMStruct1GetRadius(const struct IAMStruct1 *s)
    __attribute__((language_name("getter:Struct1.radius(self:)")));

extern void IAMStruct1SetRadius(struct IAMStruct1 s, double radius)
    __attribute__((language_name("setter:Struct1.radius(self:_:)")));

extern double IAMStruct1GetAltitude(struct IAMStruct1 s)
    __attribute__((language_name("getter:Struct1.altitude(self:)")));

extern void IAMStruct1SetAltitude(struct IAMStruct1 *s, double altitude)
    __attribute__((language_name("setter:Struct1.altitude(self:_:)")));

extern double IAMStruct1GetMagnitude(struct IAMStruct1 s)
    __attribute__((language_name("getter:Struct1.magnitude(self:)")));

extern int IAMStruct1StaticMethod(void)
    __attribute__((language_name("Struct1.staticMethod()")));
extern int IAMStruct1StaticGetProperty(void)
    __attribute__((language_name("getter:Struct1.property()")));
extern int IAMStruct1StaticSetProperty(int i)
    __attribute__((language_name("setter:Struct1.property(i:)")));
extern int IAMStruct1StaticGetOnlyProperty(void)
    __attribute__((language_name("getter:Struct1.getOnlyProperty()")));

extern void IAMStruct1SelfComesLast(double x, struct IAMStruct1 s)
    __attribute__((language_name("Struct1.selfComesLast(x:self:)")));
extern void IAMStruct1SelfComesThird(int a, float b, struct IAMStruct1 s, double x)
    __attribute__((language_name("Struct1.selfComesThird(a:b:self:x:)")));

#endif // IMPORT_AS_MEMBER_H
