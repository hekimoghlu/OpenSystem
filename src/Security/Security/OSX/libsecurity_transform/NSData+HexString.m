/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 10, 2024.
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
//  NSData+HexString.m
//  libsecurity_transform
//
//  Copyright (c) 2011,2014 Apple Inc. All Rights Reserved.
//

#import "NSData+HexString.h"

@implementation NSData (HexString)

// Not efficent
+(id)dataWithHexString:(NSString *)hex
{
	char buf[3];
	buf[2] = '\0';
	NSAssert(0 == [hex length] % 2, @"Hex strings should have an even number of digits (%@)", hex);
	unsigned char *bytes = malloc([hex length]/2);
	unsigned char *bp = bytes;
	for (CFIndex i = 0; i < [hex length]; i += 2) {
		buf[0] = [hex characterAtIndex:i];
		buf[1] = [hex characterAtIndex:i+1];
		char *b2 = NULL;
		*bp++ = strtol(buf, &b2, 16);
		NSAssert(b2 == buf + 2, @"String should be all hex digits: %@ (bad digit around %ld)", hex, i);
	}
	
	return [NSData dataWithBytesNoCopy:bytes length:[hex length]/2 freeWhenDone:YES];
}

@end
