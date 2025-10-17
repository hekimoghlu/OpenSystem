/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 29, 2023.
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
#import <Foundation/Foundation.h>

@implementation NSCoder (pyobjc)

-(void)__pyobjc__encodeInt:(int)value
{
	[self encodeValueOfObjCType:"i" at:&value];
}

-(void)__pyobjc__encodeInt32:(int)value
{
	[self encodeValueOfObjCType:"i" at:&value];
}

-(void)__pyobjc__encodeInt64:(long long)value
{
	[self encodeValueOfObjCType:"q" at:&value];
}

-(void)__pyobjc__encodeBool:(bool)value
{
	[self encodeValueOfObjCType:"b" at:&value];
}

-(int)__pyobjc__decodeInt
{
	int value;
	[self decodeValueOfObjCType:"i" at:&value];
	return value;
}

-(int)__pyobjc__decodeInt32
{
	int value;
	[self decodeValueOfObjCType:"i" at:&value];
	return value;
}

-(long long)__pyobjc__decodeInt64
{
	long long value;
	[self decodeValueOfObjCType:"q" at:&value];
	return value;
}

-(bool)__pyobjc__decodeBool
{
	bool value;
	[self decodeValueOfObjCType:"b" at:&value];
	return value;
}

@end
