/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 4, 2022.
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
#import "NSDictionary+compactDescription.h"
#import "NSString+compactDescription.h"

@implementation NSDictionary (compactDescription)

-(NSString*)compactDescription
{
	NSMutableArray *results = [NSMutableArray new];
	for (NSString *k in self) {
		id v = self[k];
		if ([v respondsToSelector:@selector(compactDescription)]) {
			v = [v compactDescription];
		} else {
			v = [v description];
		}
		
		[results addObject:[NSString stringWithFormat:@"%@=%@", [k compactDescription], v]];
	}
	return [NSString stringWithFormat:@"{%@}", [results componentsJoinedByString:@", "]];
}

-(NSString*)compactDescriptionWithoutItemData
{
	NSMutableArray *results = [NSMutableArray new];
	for (NSString *k in self) {
		if ([k isEqualToString:(__bridge NSString*) kSecValueData]) {
			[results addObject:[NSString stringWithFormat:@"%@=<not-logged>", [k compactDescription]]];
			continue;
		}
		
		id v = self[k];
		if ([v respondsToSelector:@selector(compactDescription)]) {
			v = [v compactDescription];
		} else {
			v = [v description];
		}
		
		[results addObject:[NSString stringWithFormat:@"%@=%@", [k compactDescription], v]];
	}
	return [NSString stringWithFormat:@"{%@}", [results componentsJoinedByString:@", "]];

}

@end
