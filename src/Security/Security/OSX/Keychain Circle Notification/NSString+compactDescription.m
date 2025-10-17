/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 26, 2021.
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
#import "NSString+compactDescription.h"

@implementation NSString (compactDescription)

-(NSString*)compactDescription
{
	static NSCharacterSet *forceQuotes = nil;
	static dispatch_once_t setup;
	dispatch_once(&setup, ^{
		forceQuotes = [NSCharacterSet characterSetWithCharactersInString:@"\"' \t\n\r="];
	});
	
	if ([self rangeOfCharacterFromSet:forceQuotes].location != NSNotFound) {
		NSString *escaped = [self stringByReplacingOccurrencesOfString:@"\\" withString:@"\\\\"];
		escaped = [escaped stringByReplacingOccurrencesOfString:@"\"" withString:@"\\\""];
		return [NSString stringWithFormat:@"\"%@\"", escaped];
	} else {
		return self;
	}
}

@end
