/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 15, 2022.
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
#if PLATFORM(IOS_FAMILY)

#import "WebSelectionRect.h"

@implementation WebSelectionRect {
    CGRect m_rect;
    WKWritingDirection m_writingDirection;
    BOOL m_isLineBreak;
    BOOL m_isFirstOnLine;
    BOOL m_isLastOnLine;
    BOOL m_containsStart;
    BOOL m_containsEnd;
    BOOL m_isInFixedPosition;
    BOOL m_isHorizontal;
}

@synthesize rect = m_rect;
@synthesize writingDirection = m_writingDirection;
@synthesize isLineBreak = m_isLineBreak;
@synthesize isFirstOnLine = m_isFirstOnLine;
@synthesize isLastOnLine = m_isLastOnLine;
@synthesize containsStart = m_containsStart;
@synthesize containsEnd = m_containsEnd;
@synthesize isInFixedPosition = m_isInFixedPosition;
@synthesize isHorizontal = m_isHorizontal;

+ (WebSelectionRect *)selectionRect
{
    return [[(WebSelectionRect *)[self alloc] init] autorelease];
}

+ (CGRect)startEdge:(NSArray *)rects
{
    if (rects.count == 0)
        return CGRectZero;

    WebSelectionRect *selectionRect = nil;
    for (WebSelectionRect *srect in rects) {
        if (srect.containsStart) {
            selectionRect = srect;
            break;
        }
    }
    if (!selectionRect)
        selectionRect = [rects objectAtIndex:0];
    
    CGRect rect = selectionRect.rect;
    
    if (!selectionRect.isHorizontal) {
        switch (selectionRect.writingDirection) {
            case WKWritingDirectionNatural:
            case WKWritingDirectionLeftToRight:
                // collapse to top edge
                rect.size.height = 1;
                break;
            case WKWritingDirectionRightToLeft:
                // collapse to bottom edge
                rect.origin.y += (rect.size.height - 1);
                rect.size.height = 1;
                break;
        }
        return rect;
    }
    
    switch (selectionRect.writingDirection) {
        case WKWritingDirectionNatural:
        case WKWritingDirectionLeftToRight:
            // collapse to left edge
            rect.size.width = 1;
            break;
        case WKWritingDirectionRightToLeft:
            // collapse to right edge
            rect.origin.x += (rect.size.width - 1);
            rect.size.width = 1;
            break;
    }
    return rect;
}

+ (CGRect)endEdge:(NSArray *)rects
{
    if (rects.count == 0)
        return CGRectZero;

    WebSelectionRect *selectionRect = nil;
    for (WebSelectionRect *srect in rects) {
        if (srect.containsEnd) {
            selectionRect = srect;
            break;
        }
    }
    if (!selectionRect)
        selectionRect = [rects lastObject];

    CGRect rect = selectionRect.rect;
    
    if (!selectionRect.isHorizontal) {
        switch (selectionRect.writingDirection) {
            case WKWritingDirectionNatural:
            case WKWritingDirectionLeftToRight:
                // collapse to bottom edge
                rect.origin.y += (rect.size.height - 1);
                rect.size.height = 1;
                break;
            case WKWritingDirectionRightToLeft:
                // collapse to top edge
                rect.size.height = 1;
                break;
        }
        return rect;
    }
    
    switch (selectionRect.writingDirection) {
        case WKWritingDirectionNatural:
        case WKWritingDirectionLeftToRight:
            // collapse to right edge
            rect.origin.x += (rect.size.width - 1);
            rect.size.width = 1;
            break;
        case WKWritingDirectionRightToLeft:
            // collapse to left edge
            rect.size.width = 1;
            break;
    }
    return rect;
}

- (id)init
{
    self = [super init];
    if (!self)
        return nil;
    
    self.rect = CGRectZero;
    self.writingDirection = WKWritingDirectionLeftToRight;
    self.isLineBreak = NO;
    self.isFirstOnLine = NO;
    self.isLastOnLine = NO;
    self.containsStart = NO;
    self.containsEnd = NO;
    self.isInFixedPosition = NO;
    self.isHorizontal = NO;
    
    return self;
}

- (id)copyWithZone:(NSZone *)zone
{
    WebSelectionRect *copy = [[WebSelectionRect selectionRect] retain];
    copy.rect = self.rect;
    copy.writingDirection = self.writingDirection;
    copy.isLineBreak = self.isLineBreak;
    copy.isFirstOnLine = self.isFirstOnLine;
    copy.isLastOnLine = self.isLastOnLine;
    copy.containsStart = self.containsStart;
    copy.containsEnd = self.containsEnd;
    copy.isInFixedPosition = self.isInFixedPosition;
    copy.isHorizontal = self.isHorizontal;
    return copy;
}

- (NSString *)description
{
    return [NSString stringWithFormat:@"<WebSelectionRect: %p> : { %.1f,%.1f,%.1f,%.1f } %@%@%@%@%@%@%@%@",
        self, 
        self.rect.origin.x, self.rect.origin.y, self.rect.size.width, self.rect.size.height,
        self.writingDirection == WKWritingDirectionLeftToRight ? @"[LTR]" : @"[RTL]",
        self.isLineBreak ? @" [BR]" : @"",
        self.isFirstOnLine ? @" [FIRST]" : @"",
        self.isLastOnLine ? @" [LAST]" : @"",
        self.containsStart ? @" [START]" : @"",
        self.containsEnd ? @" [END]" : @"",
        self.isInFixedPosition ? @" [FIXED]" : @"",
        !self.isHorizontal ? @" [VERTICAL]" : @""];
}

@end

#endif  // PLATFORM(IOS_FAMILY)
