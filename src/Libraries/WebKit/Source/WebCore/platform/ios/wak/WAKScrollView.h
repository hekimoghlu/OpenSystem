/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 18, 2025.
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
#ifndef WAKScrollView_h
#define WAKScrollView_h

#if TARGET_OS_IPHONE

#import "WAKView.h"
#import "WebCoreFrameView.h"
#import <Foundation/Foundation.h>

@class WAKClipView;

WEBCORE_EXPORT @interface WAKScrollView : WAKView <WebCoreFrameScrollView>
{
    WAKView *_documentView;  // Only here so the ObjC instance stays around.
    WAKClipView *_contentView;
    NSPoint _scrollOrigin;
}

- (CGRect)documentVisibleRect;
- (WAKClipView *)contentView;
- (id)documentView;
- (void)setDocumentView:(WAKView *)aView;
- (void)setHasVerticalScroller:(BOOL)flag;
- (BOOL)hasVerticalScroller;
- (void)setHasHorizontalScroller:(BOOL)flag;
- (BOOL)hasHorizontalScroller;
- (void)reflectScrolledClipView:(WAKClipView *)aClipView;
- (void)setDrawsBackground:(BOOL)flag;
- (float)verticalLineScroll;
- (void)setLineScroll:(float)aFloat;
- (BOOL)drawsBackground;
- (float)horizontalLineScroll;

@property (nonatomic, weak) id delegate;

- (CGRect)unobscuredContentRect;
- (void)setActualScrollPosition:(CGPoint)point;

// Like unobscuredContentRect, but includes areas possibly covered by translucent UI.
- (CGRect)exposedContentRect;

- (BOOL)inProgrammaticScroll;
@end

@interface NSObject (WAKScrollViewDelegate)
- (BOOL)scrollView:(WAKScrollView *)scrollView shouldScrollToPoint:(CGPoint)point;
@end

#endif // TARGET_OS_IPHONE

#endif // WAKScrollView_h
