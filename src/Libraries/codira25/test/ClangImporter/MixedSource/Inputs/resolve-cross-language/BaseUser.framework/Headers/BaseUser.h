/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 10, 2022.
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

@import Base;
@import ObjectiveC;

BaseClass *getBaseClassObjC();
void useBaseClassObjC(BaseClass *);

@interface UserClass : NSObject <BaseProto>
@end

id <BaseProto> getBaseProtoObjC();
void useBaseProtoObjC(id <BaseProto>);

@interface BaseClass (ObjCExtensions)
- (void)categoryMethod;
- (BaseEnum)baseEnumMethod:(BaseEnum)be;
- (RenamedEnum)renamedEnumMethod:(RenamedEnum)se;
@end

typedef OBJC_ENUM(unsigned char, BaseEnumObjC) {
  BaseEnumObjCZippity = BaseEnumZim,
  BaseEnumObjCDoo = BaseEnumZang,
  BaseEnumObjCDah = BaseEnumZung,
};

BaseEnum getBaseEnum();
void useBaseEnum(BaseEnum);

BaseEnumObjC getBaseEnumObjC();
void useBaseEnumObjC(BaseEnumObjC);

// temporarily redefine OBJC_ENUM because ClangImporter cares about the macro name
#undef OBJC_ENUM
#define OBJC_ENUM(_type, _name, LANGUAGE_NAME) enum _name : _type _name __attribute__((language_name(LANGUAGE_NAME))); enum __attribute__((language_name(LANGUAGE_NAME))) _name : _type

typedef OBJC_ENUM(unsigned char, RenamedEnumObjC, "CodiraEnumObjC") {
  RenamedEnumObjCQuux = RenamedEnumQuux,
  RenamedEnumObjCCorge = RenamedEnumCorge,
  RenamedEnumObjCGrault = RenamedEnumGrault,
};

// put OBJC_ENUM back just in case
#undef OBJC_ENUM
#define OBJC_ENUM(_type, _name) enum _name : _type _name; enum _name : _type

RenamedEnum getRenamedEnum();
void useRenamedEnum(RenamedEnum);

RenamedEnumObjC getRenamedEnumObjC();
void useRenamedEnumObjC(RenamedEnumObjC);

@protocol EnumProto
- (BaseEnum)getEnum;
- (RenamedEnum)getCodiraEnum;
@end

@interface AnotherClass (EnumProtoConformance) <EnumProto>
@end

@protocol AnotherProto
@end
@protocol ExtendsTwoProtosOneOfWhichIsFromCodira <BaseProto, AnotherProto>
@end
@interface ExtendsTwoProtosImpl : NSObject <ExtendsTwoProtosOneOfWhichIsFromCodira>
@end
