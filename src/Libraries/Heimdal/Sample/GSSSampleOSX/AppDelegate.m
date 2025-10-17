/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 31, 2023.
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
//  AppDelegate.m
//  GSSSampleOSX
//
//  Created by Love HÃ¶rnquist Ã…strand on 2011-11-13.
//

#import "AppDelegate.h"
#import <GSS/GSSItem.h>

@implementation AppDelegate

@synthesize window = _window;
@synthesize tableview = _tableview;
@synthesize credentials = _credentials;
@synthesize arrayController = _arrayController;

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification
{
	[self refreshCredentials:nil];
}

- (IBAction)refreshCredentials:(id)sender
{
	_credentials = [[NSMutableArray alloc] init];
	
	CFMutableDictionaryRef attrs = CFDictionaryCreateMutable(NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
	
	CFDictionaryAddValue(attrs, kGSSAttrClass, kGSSAttrClassKerberos);
	
	CFErrorRef error = NULL;
	
	CFArrayRef items = GSSItemCopyMatching(attrs, &error);
	if (items) {
		CFIndex n, count = CFArrayGetCount(items);
		for (n = 0; n < count; n++) {
			CFTypeRef item = CFArrayGetValueAtIndex(items, n);
			NSLog(@"item %d = %@", (int)n, item);
			
			NSDictionary *i;
			
			i = [(__bridge NSDictionary *)item mutableCopy];
			[i setValue:@"expire1" forKey:@"kGSSAttrTransientExpire"];
			NSLog(@"%@ %@", i, [i className]);
			[_credentials addObject:i];
		}
		CFRelease(items);
	}
	CFRelease(attrs);
	
	[_credentials addObject:@{ @"kGSSAttrNameDisplay" : @"foo", @"kGSSAttrTransientExpire" : @"expire"}];
	
	NSLog(@"%@", _credentials);
	
	[_arrayController setContent:_credentials];
	
	NSLog(@"item %@", [_arrayController valueForKeyPath:@"arrangedObjects.kGSSAttrNameDisplay"]);
	NSLog(@"item %@", [_arrayController valueForKeyPath:@"arrangedObjects.kGSSAttrTransientExpire"]);
	
	[_tableview reloadData];
	
	
}
@end
