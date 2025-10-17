/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 13, 2023.
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
#import "WebKitNSStringExtras.h"

#import <WebCore/ColorMac.h>
#import <WebCore/FontCascade.h>
#import <WebCore/GraphicsContextCG.h>
#import <WebCore/LoaderNSURLExtras.h>
#import <WebCore/TextRun.h>
#import <pal/spi/cg/CoreGraphicsSPI.h>
#import <sys/param.h>
#import <unicode/uchar.h>

NSString *WebKitLocalCacheDefaultsKey = @"WebKitLocalCache";
NSString *WebKitResourceLoadStatisticsDirectoryDefaultsKey = @"WebKitResourceLoadStatisticsDirectory";

using namespace WebCore;

@implementation NSString (WebKitExtras)

#if PLATFORM(MAC)

static bool canUseFastRenderer(std::span<const UniChar> buffer)
{
    for (auto character : buffer) {
        if (!isLatin1(character)) {
            auto direction = u_charDirection(character);
            if (direction == U_RIGHT_TO_LEFT || (direction > U_OTHER_NEUTRAL && direction != U_DIR_NON_SPACING_MARK && direction != U_BOUNDARY_NEUTRAL))
                return false;
        }
    }
    return true;
}

- (void)_web_drawAtPoint:(NSPoint)point font:(NSFont *)font textColor:(NSColor *)textColor
{
    if (!font)
        return;

    unsigned length = [self length];
    Vector<UniChar, 2048> buffer(length);
    [self getCharacters:buffer.data()];

    if (canUseFastRenderer(buffer)) {
        FontCascade webCoreFont(FontPlatformData((__bridge CTFontRef)font, [font pointSize]));
        TextRun run(StringView(spanReinterpretCast<const UChar>(std::span { buffer })));

        // The following is a half-assed attempt to match AppKit's rounding rules for drawAtPoint.
        // If you change it, be sure to test all the text drawn this way in Safari, including
        // the status bar, bookmarks bar, tab bar, and activity window.
        point.y = CGCeiling(point.y);

        NSGraphicsContext *nsContext = [NSGraphicsContext currentContext];
        CGContextRef cgContext = [nsContext CGContext];
        GraphicsContextCG graphicsContext { cgContext };

        // WebCore requires a flipped graphics context.
        bool flipped = [nsContext isFlipped];
        if (!flipped)
            CGContextScaleCTM(cgContext, 1, -1);

        graphicsContext.setFillColor(colorFromCocoaColor(textColor));
        webCoreFont.drawText(graphicsContext, run, FloatPoint(point.x, flipped ? point.y : -point.y));

        if (!flipped)
            CGContextScaleCTM(cgContext, 1, -1);
    } else {
        // The given point is on the baseline.
        if ([[NSView focusView] isFlipped])
            point.y -= [font ascender];
        else
            point.y += [font descender];

        [self drawAtPoint:point withAttributes:@{ NSFontAttributeName: font, NSForegroundColorAttributeName: textColor }];
    }
}

- (float)_web_widthWithFont:(NSFont *)font
{
    unsigned length = [self length];
    Vector<UniChar, 2048> buffer(length);
    [self getCharacters:buffer.data()];

    if (canUseFastRenderer(buffer)) {
        FontCascade webCoreFont(FontPlatformData((__bridge CTFontRef)font, [font pointSize]));
        TextRun run(StringView(spanReinterpretCast<const UChar>(std::span { buffer })));
        return webCoreFont.width(run);
    }

    return [self sizeWithAttributes:@{ NSFontAttributeName: font }].width;
}

#endif // PLATFORM(MAC)

- (NSString *)_web_stringByAbbreviatingWithTildeInPath
{
    // Handles home directories that have symlinks in their paths as well as what stringByAbbreviatingWithTildeInPath handles.
    // This works around Radar bug 2774250.

    NSString *resolvedHomeDirectory = [NSHomeDirectory() stringByResolvingSymlinksInPath];
    NSString *path;

    if ([self hasPrefix:resolvedHomeDirectory]) {
        NSString *relativePath = [self substringFromIndex:[resolvedHomeDirectory length]];
        path = [NSHomeDirectory() stringByAppendingPathComponent:relativePath];
    } else {
        path = self;
    }

    return [path stringByAbbreviatingWithTildeInPath];
}

- (BOOL)_webkit_isCaseInsensitiveEqualToString:(NSString *)string
{
    return [self compare:string options:(NSCaseInsensitiveSearch | NSLiteralSearch)] == NSOrderedSame;
}

-(BOOL)_webkit_hasCaseInsensitivePrefix:(NSString *)prefix
{
    return [self rangeOfString:prefix options:(NSCaseInsensitiveSearch | NSAnchoredSearch)].location != NSNotFound;
}

-(BOOL)_webkit_hasCaseInsensitiveSuffix:(NSString *)suffix
{
    return [self rangeOfString:suffix options:(NSCaseInsensitiveSearch | NSBackwardsSearch | NSAnchoredSearch)].location != NSNotFound;
}

-(NSString *)_webkit_filenameByFixingIllegalCharacters
{
    return filenameByFixingIllegalCharacters(self);
}

-(NSString *)_webkit_stringByTrimmingWhitespace
{
    auto trimmed = adoptNS([self mutableCopy]);
    CFStringTrimWhitespace((__bridge CFMutableStringRef)trimmed.get());
    return trimmed.autorelease();
}

+ (NSString *)_webkit_localCacheDirectoryWithBundleIdentifier:(NSString*)bundleIdentifier
{
    NSString *cacheDirectory = [[NSUserDefaults standardUserDefaults] objectForKey:WebKitLocalCacheDefaultsKey];

    if (!cacheDirectory || ![cacheDirectory isKindOfClass:[NSString class]]) {
#if PLATFORM(IOS_FAMILY)
        cacheDirectory = [NSHomeDirectory() stringByAppendingPathComponent:@"Library/Caches"];
#endif
#if PLATFORM(MAC)
        char buffer[MAXPATHLEN];
        if (size_t length = confstr(_CS_DARWIN_USER_CACHE_DIR, buffer, MAXPATHLEN))
            cacheDirectory = [[NSFileManager defaultManager] stringWithFileSystemRepresentation:buffer length:length - 1];
#endif
    }

    return [cacheDirectory stringByAppendingPathComponent:bundleIdentifier];
}

@end
