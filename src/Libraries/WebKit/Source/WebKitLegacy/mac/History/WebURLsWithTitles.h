/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 3, 2022.
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
#import <TargetConditionals.h>

#if !TARGET_OS_IPHONE

#import <Cocoa/Cocoa.h>

#define WebURLsWithTitlesPboardType     @"WebURLsWithTitlesPboardType"

// Convenience class for getting URLs and associated titles on and off an NSPasteboard

@interface WebURLsWithTitles : NSObject

// Writes parallel arrays of URLs and titles to the pasteboard. These items can be retrieved by
// calling URLsFromPasteboard and titlesFromPasteboard. URLs must consist of NSURL objects.
// titles must consist of NSStrings, or be nil. If titles is nil, or if titles is a different
// length than URLs, empty strings will be used for all titles. If URLs is nil, this method
// returns without doing anything. You must declare an WebURLsWithTitlesPboardType data type
// for pasteboard before invoking this method, or it will return without doing anything.
+ (void)writeURLs:(NSArray *)URLs andTitles:(NSArray *)titles toPasteboard:(NSPasteboard *)pasteboard;

// Reads an array of NSURLs off the pasteboard. Returns nil if pasteboard does not contain
// data of type WebURLsWithTitlesPboardType. This array consists of the URLs that correspond to
// the titles returned from titlesFromPasteboard.
+ (NSArray *)URLsFromPasteboard:(NSPasteboard *)pasteboard;

// Reads an array of NSStrings off the pasteboard. Returns nil if pasteboard does not contain
// data of type WebURLsWithTitlesPboardType. This array consists of the titles that correspond to
// the URLs returned from URLsFromPasteboard.
+ (NSArray *)titlesFromPasteboard:(NSPasteboard *)pasteboard;

@end

#endif
