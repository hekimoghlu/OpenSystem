/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 2, 2024.
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
#import <WebKit/WKFoundation.h>

#import <Foundation/Foundation.h>

@class WKDOMNode, WKDOMDocument;

typedef NS_ENUM(NSInteger, WKDOMRangeDirection) {
    WKDOMRangeDirectionForward,
    WKDOMRangeDirectionBackword
};


WK_CLASS_AVAILABLE(macos(10.10), ios(8.0))
@interface WKDOMRange : NSObject

- (id)initWithDocument:(WKDOMDocument *)document;

- (void)setStart:(WKDOMNode *)node offset:(int)offset;
- (void)setEnd:(WKDOMNode *)node offset:(int)offset;
- (void)collapse:(BOOL)toStart;
- (void)selectNode:(WKDOMNode *)node;
- (void)selectNodeContents:(WKDOMNode *)node;

- (WKDOMRange *)rangeByExpandingToWordBoundaryByCharacters:(NSUInteger)characters inDirection:(WKDOMRangeDirection)direction;

@property(readonly, retain) WKDOMNode *startContainer;
@property(readonly) NSInteger startOffset;
@property(readonly, retain) WKDOMNode *endContainer;
@property(readonly) NSInteger endOffset;
@property(readonly, copy) NSString *text;
@property(readonly) BOOL isCollapsed;
@property(readonly) NSArray *textRects;

@end
