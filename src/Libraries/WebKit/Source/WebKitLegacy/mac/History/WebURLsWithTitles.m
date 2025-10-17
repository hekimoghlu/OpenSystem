/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 12, 2025.
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
#if !PLATFORM(IOS_FAMILY)

#import "WebURLsWithTitles.h"

#import <WebKitLegacy/WebNSURLExtras.h>
#import <WebKitLegacy/WebKitNSStringExtras.h>

@implementation WebURLsWithTitles

+ (NSArray *)arrayWithIFURLsWithTitlesPboardType
{
    // Make a canned array so we don't construct it on the fly over and over.
    static NSArray *cannedArray = nil;

    if (cannedArray == nil) {
        cannedArray = [@[WebURLsWithTitlesPboardType] retain];
    }

    return cannedArray;
}

+(void)writeURLs:(NSArray *)URLs andTitles:(NSArray *)titles toPasteboard:(NSPasteboard *)pasteboard
{
    NSMutableArray *URLStrings;
    NSMutableArray *titlesOrEmptyStrings;
    NSUInteger index, count;

    count = [URLs count];
    if (count == 0) {
        return;
    }

    if ([pasteboard availableTypeFromArray:[self arrayWithIFURLsWithTitlesPboardType]] == nil) {
        return;
    }

    if (count != [titles count]) {
        titles = nil;
    }

    URLStrings = [NSMutableArray arrayWithCapacity:count];
    titlesOrEmptyStrings = [NSMutableArray arrayWithCapacity:count];
    for (index = 0; index < count; ++index) {
        [URLStrings addObject:[[URLs objectAtIndex:index] _web_originalDataAsString]];
        [titlesOrEmptyStrings addObject:(titles == nil) ? @"" : [[titles objectAtIndex:index] _webkit_stringByTrimmingWhitespace]];
    }

    [pasteboard setPropertyList:@[URLStrings, titlesOrEmptyStrings]
                        forType:WebURLsWithTitlesPboardType];
}

+(NSArray *)titlesFromPasteboard:(NSPasteboard *)pasteboard
{
    if ([pasteboard availableTypeFromArray:[self arrayWithIFURLsWithTitlesPboardType]] == nil) {
        return nil;
    }

    return [[pasteboard propertyListForType:WebURLsWithTitlesPboardType] objectAtIndex:1];
}

+(NSArray *)URLsFromPasteboard:(NSPasteboard *)pasteboard
{
    NSArray *URLStrings;
    NSMutableArray *URLs;
    unsigned index, count;
    
    if ([pasteboard availableTypeFromArray:[self arrayWithIFURLsWithTitlesPboardType]] == nil) {
        return nil;
    }

    URLStrings = [[pasteboard propertyListForType:WebURLsWithTitlesPboardType] objectAtIndex:0];
    count = [URLStrings count];
    URLs = [NSMutableArray arrayWithCapacity:count];
    for (index = 0; index < count; ++index) {
        [URLs addObject:[NSURL _web_URLWithDataAsString:[URLStrings objectAtIndex:index]]];
    }

    return URLs;
}

@end

#endif // !PLATFORM(IOS_FAMILY)
