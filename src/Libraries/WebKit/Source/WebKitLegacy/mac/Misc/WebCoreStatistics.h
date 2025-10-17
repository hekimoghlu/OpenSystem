/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 25, 2025.
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
#import <CoreGraphics/CGContext.h>
#import <Foundation/Foundation.h>
#import <WebKitLegacy/WebFrame.h>

@class DOMElement;

@interface WebCoreStatistics : NSObject
{
}

+ (NSArray *)statistics;

+ (size_t)javaScriptObjectsCount;
+ (size_t)javaScriptGlobalObjectsCount;
+ (size_t)javaScriptProtectedObjectsCount;
+ (size_t)javaScriptProtectedGlobalObjectsCount;
+ (NSCountedSet *)javaScriptProtectedObjectTypeCounts;
+ (NSCountedSet *)javaScriptObjectTypeCounts;

+ (void)garbageCollectJavaScriptObjects;
+ (void)garbageCollectJavaScriptObjectsOnAlternateThreadForDebugging:(BOOL)waitUntilDone;
+ (void)setJavaScriptGarbageCollectorTimerEnabled:(BOOL)enabled;

+ (size_t)iconPageURLMappingCount;
+ (size_t)iconRetainedPageURLCount;
+ (size_t)iconRecordCount;
+ (size_t)iconsWithDataCount;

+ (size_t)cachedFontDataCount;
+ (size_t)cachedFontDataInactiveCount;
+ (void)purgeInactiveFontData;
+ (size_t)glyphPageCount;

+ (BOOL)shouldPrintExceptions;
+ (void)setShouldPrintExceptions:(BOOL)print;

+ (void)startIgnoringWebCoreNodeLeaks;
+ (void)stopIgnoringWebCoreNodeLeaks;

+ (NSDictionary *)memoryStatistics;
+ (void)returnFreeMemoryToSystem;

+ (int)cachedPageCount;
+ (int)cachedFrameCount;
+ (int)autoreleasedPageCount;

// Deprecated, but used by older versions of Safari.
+ (void)emptyCache;
+ (void)setCacheDisabled:(BOOL)disabled;
+ (size_t)javaScriptNoGCAllowedObjectsCount;
+ (size_t)javaScriptReferencedObjectsCount;
+ (NSSet *)javaScriptRootObjectClasses;
+ (NSCountedSet *)javaScriptRootObjectTypeCounts;
+ (size_t)javaScriptInterpretersCount;

@end

typedef NS_OPTIONS(NSUInteger, WebRenderTreeAsTextOptions) {
    WebRenderTreeAsTextShowAllLayers           = 1 << 0,
    WebRenderTreeAsTextShowLayerNesting        = 1 << 1,
    WebRenderTreeAsTextShowCompositedLayers    = 1 << 2,
    WebRenderTreeAsTextShowOverflow            = 1 << 3,
    WebRenderTreeAsTextShowSVGGeometry         = 1 << 4,
    WebRenderTreeAsTextShowLayerFragments      = 1 << 5
};

@interface WebFrame (WebKitDebug)
- (NSString *)renderTreeAsExternalRepresentationForPrinting;
- (NSString *)renderTreeAsExternalRepresentationWithOptions:(WebRenderTreeAsTextOptions)options;
- (int)numberOfPagesWithPageWidth:(float)pageWidthInPixels pageHeight:(float)pageHeightInPixels;
- (void)printToCGContext:(CGContextRef)cgContext pageWidth:(float)pageWidthInPixels pageHeight:(float)pageHeightInPixels;
@end
