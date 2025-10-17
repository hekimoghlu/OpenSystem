/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 13, 2023.
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
#if USE(APPKIT)

#if USE(APPLE_INTERNAL_SDK)

#import <AppKit/NSTextFinder_Private.h>

#else

typedef enum : NSUInteger {
    NSTextFinderAsynchronousDocumentFindOptionsBackwards = 1 << 0,
    NSTextFinderAsynchronousDocumentFindOptionsWrap = 1 << 1,
    NSTextFinderAsynchronousDocumentFindOptionsCaseInsensitive = 1 << 2,
    NSTextFinderAsynchronousDocumentFindOptionsStartsWith = 1 << 3,
} NSTextFinderAsynchronousDocumentFindOptions;

@protocol NSTextFinderAsynchronousDocumentFindMatch <NSObject>
@required
@property (retain, nonatomic, readonly) NSView *containingView;
@property (retain, nonatomic, readonly) NSArray *textRects;
- (void)generateTextImage:(void (^)(NSImage *))completionHandler;
@end

@interface NSObject ()

- (void)findMatchesForString:(NSString *)targetString relativeToMatch:(id <NSTextFinderAsynchronousDocumentFindMatch>)relativeMatch findOptions:(NSTextFinderAsynchronousDocumentFindOptions)findOptions maxResults:(NSUInteger)maxResults resultCollector:(void (^)(NSArray *matches, BOOL didWrap))resultCollector;
- (NSView *)documentContainerView;
- (void)getSelectedText:(void (^)(NSString *selectedTextString))completionHandler;
- (void)selectFindMatch:(id <NSTextFinderAsynchronousDocumentFindMatch>)findMatch completionHandler:(void (^)(void))completionHandler;
- (void)replaceMatches:(NSArray *)matches withString:(NSString *)replacementString inSelectionOnly:(BOOL)selectionOnly resultCollector:(void (^)(NSUInteger replacementCount))resultCollector;
- (void)scrollFindMatchToVisible:(id <NSTextFinderAsynchronousDocumentFindMatch>)findMatch;

@end

#endif // USE(APPLE_INTERNAL_SDK)

#endif // USE(APPKIT)
