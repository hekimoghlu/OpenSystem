/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 4, 2024.
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


#if __has_include(<Foundation/Foundation.h>)

#import "NSSlowString.h"


@interface NSSlowString ()

@property (nonatomic, strong) NSString *stringHolder;

@end

@implementation NSSlowString

- (instancetype)initWithString:(NSString *)name {
	self = [super init];
	if (self == nil) {
		return nil;
	}
	self.stringHolder = name;
	return self;
}

- (instancetype)initWithCharacters:(const unichar * _Nonnull)chars length:(NSUInteger)count {
  NSString *str = [[NSString alloc] initWithCharacters: chars length: count];
  self = [self initWithString: str];
  return self;
}

- (NSUInteger)length {
    return self.stringHolder.length;
}

- (id)copyWithZone:(NSZone *)unused {
	return self;
}

- (unichar)characterAtIndex:(NSUInteger)index {
    return [self.stringHolder characterAtIndex:index];
}

- (void *) _fastCharacterContents {
  return nil;
}

@end

#endif

