/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 23, 2023.
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

#import <WebKitLegacy/WebNSImageExtras.h>

#import <WebKitLegacy/WebKitLogging.h>

@implementation NSImage (WebExtras)

- (void)_web_scaleToMaxSize:(NSSize)size
{
    float heightResizeDelta = 0.0f, widthResizeDelta = 0.0f, resizeDelta = 0.0f;
    NSSize originalSize = [self size];

    if(originalSize.width > size.width){
        widthResizeDelta = size.width / originalSize.width;
        resizeDelta = widthResizeDelta;
    }

    if(originalSize.height > size.height){
        heightResizeDelta = size.height / originalSize.height;
        if((resizeDelta == 0.0) || (resizeDelta > heightResizeDelta)){
            resizeDelta = heightResizeDelta;
        }
    }
    
    if(resizeDelta > 0.0){
        NSSize newSize = NSMakeSize((originalSize.width * resizeDelta), (originalSize.height * resizeDelta));
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
        [self setScalesWhenResized:YES];
ALLOW_DEPRECATED_DECLARATIONS_END
        [self setSize:newSize];
    }
}

- (void)_web_dissolveToFraction:(float)delta
{
    NSImage *dissolvedImage = [[NSImage alloc] initWithSize:[self size]];

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    NSPoint point = [self isFlipped] ? NSMakePoint(0, [self size].height) : NSZeroPoint;
ALLOW_DEPRECATED_DECLARATIONS_END
    
    // In this case the dragging image is always correct.
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    [dissolvedImage setFlipped:[self isFlipped]];
ALLOW_DEPRECATED_DECLARATIONS_END

    [dissolvedImage lockFocus];
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    [self dissolveToPoint:point fraction: delta];
ALLOW_DEPRECATED_DECLARATIONS_END
    [dissolvedImage unlockFocus];

    [self lockFocus];
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    [dissolvedImage compositeToPoint:point operation:NSCompositeCopy];
ALLOW_DEPRECATED_DECLARATIONS_END
    [self unlockFocus];

    [dissolvedImage release];
}

@end

#endif // !PLATFORM(IOS_FAMILY)
