/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 18, 2024.
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
#ifndef SECURITY_SFSQL_OBJCTYPE_H
#define SECURITY_SFSQL_OBJCTYPE_H 1

#if __OBJC2__

#import <Foundation/Foundation.h>

typedef NS_ENUM(NSInteger, SFObjCTypeCode) {
    SFObjCTypeChar                = 0,  // 'c'
    SFObjCTypeShort               = 1,  // 's'
    SFObjCTypeInt                 = 2,  // 'i'
    SFObjCTypeLong                = 3,  // 'l'
    SFObjCTypeLongLong            = 4,  // 'q'
    SFObjCTypeUnsignedChar        = 5,  // 'C'
    SFObjCTypeUnsignedShort       = 6,  // 'S'
    SFObjCTypeUnsignedInt         = 7,  // 'I'
    SFObjCTypeUnsignedLong        = 8,  // 'L'
    SFObjCTypeUnsignedLongLong    = 9,  // 'Q'
    SFObjCTypeFloat               = 10, // 'f'
    SFObjCTypeDouble              = 11, // 'd'
    SFObjCTypeBool                = 12, // 'b'
    SFObjCTypeVoid                = 13, // 'v'
    SFObjCTypeCharPointer         = 14, // '*'
    SFObjCTypeObject              = 15, // '@'
    SFObjCTypeClass               = 16, // '#'
    SFObjCTypeSelector            = 17, // ':'
    SFObjCTypeArray               = 18, // '[' type ']'
    SFObjCTypeStructure           = 19, // '{' name '=' type... '}'
    SFObjCTypeUnion               = 20, // '(' name '=' type... ')'
    SFObjCTypeBitfield            = 21, // 'b' number
    SFObjCTypePointer             = 22, // '^' type
    SFObjCTypeUnknown             = 23, // '?'
};

typedef NS_ENUM(NSInteger, SFObjCTypeFlag) {
    SFObjCTypeFlagIntegerNumber       = 0x1,
    SFObjCTypeFlagFloatingPointNumber = 0x2,
    
    SFObjCTypeFlagNone                = 0x0,
    SFObjCTypeFlagNumberMask          = 0x3,
};

@interface SFObjCType : NSObject {
    SFObjCTypeCode _code;
    NSString* _encoding;
    NSString* _name;
    NSString* _className;
    NSUInteger _size;
    NSUInteger _flags;
}

+ (SFObjCType *)typeForEncoding:(const char *)encoding;
+ (SFObjCType *)typeForValue:(NSValue *)value;

@property (nonatomic, readonly, assign) SFObjCTypeCode    code;
@property (nonatomic, readonly, strong) NSString           *encoding;
@property (nonatomic, readonly, strong) NSString           *name;
@property (nonatomic, readonly, strong) NSString           *className;
@property (nonatomic, readonly, assign) NSUInteger          size;
@property (nonatomic, readonly, assign) NSUInteger          flags;

@property (nonatomic, readonly, assign, getter=isNumber)              BOOL    number;
@property (nonatomic, readonly, assign, getter=isIntegerNumber)       BOOL    integerNumber;
@property (nonatomic, readonly, assign, getter=isFloatingPointNumber) BOOL    floatingPointNumber;
@property (nonatomic, readonly, assign, getter=isObject)              BOOL    object;

- (id)objectWithBytes:(const void *)bytes;
- (void)getBytes:(void *)bytes forObject:(id)object;

@end

#endif
#endif /* SECURITY_SFSQL_OBJCTYPE_H */
