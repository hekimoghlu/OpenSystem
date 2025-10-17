/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 11, 2024.
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
#import "config.h"
#import "Internals.h"

#import "AGXCompilerService.h"
#import "DOMURL.h"
#import "DeprecatedGlobalSettings.h"
#import "DictionaryLookup.h"
#import "Document.h"
#import "EventHandler.h"
#import "HTMLMediaElement.h"
#import "HitTestResult.h"
#import "LocalFrameView.h"
#import "MediaPlayerPrivate.h"
#import "Range.h"
#import "SharedBuffer.h"
#import "SimpleRange.h"
#import "UTIUtilities.h"
#import <AVFoundation/AVPlayer.h>

#if PLATFORM(MAC)
#import "NSScrollerImpDetails.h"
#import "ScrollbarThemeMac.h"
#import <pal/spi/mac/NSScrollerImpSPI.h>
#endif

#import <pal/spi/cocoa/NSAccessibilitySPI.h>
#import <wtf/cf/TypeCastsCF.h>
#import <wtf/cocoa/NSURLExtras.h>
#import <wtf/spi/darwin/SandboxSPI.h>
#import <wtf/unicode/CharacterNames.h>

#if PLATFORM(IOS_FAMILY)
#import <pal/ios/UIKitSoftLink.h>
#endif

#if ENABLE(DATA_DETECTION)
#import <pal/cocoa/DataDetectorsCoreSoftLink.h>
#endif

#import <pal/cocoa/VisionKitCoreSoftLink.h>

#if ENABLE(IMAGE_ANALYSIS_ENHANCEMENTS)

@interface FakeImageAnalysisResult : NSObject
- (instancetype)initWithString:(NSString *)fullText;
@end

@implementation FakeImageAnalysisResult {
    RetainPtr<NSAttributedString> _string;
}

- (instancetype)initWithString:(NSString *)string
{
    if (!(self = [super init]))
        return nil;

    _string = adoptNS([[NSMutableAttributedString alloc] initWithString:string]);
    return self;
}

- (NSAttributedString *)_attributedStringForRange:(NSRange)range
{
    return [_string attributedSubstringFromRange:range];
}

@end

#endif // ENABLE(IMAGE_ANALYSIS_ENHANCEMENTS)

#if USE(APPLE_INTERNAL_SDK) && __has_include(<WebKitAdditions/WebCoreInternalsAdditions.mm>)
#import <WebKitAdditions/WebCoreInternalsAdditions.mm>
#endif

namespace WebCore {

String Internals::userVisibleString(const DOMURL& url)
{
    return WTF::userVisibleString(url.href());
}

bool Internals::userPrefersContrast() const
{
#if PLATFORM(IOS_FAMILY)
    return PAL::softLink_UIKit_UIAccessibilityDarkerSystemColorsEnabled();
#else
    return [[NSWorkspace sharedWorkspace] accessibilityDisplayShouldIncreaseContrast];
#endif
}

bool Internals::userPrefersReducedMotion() const
{
#if PLATFORM(IOS_FAMILY)
    return PAL::softLink_UIKit_UIAccessibilityIsReduceMotionEnabled();
#else
    return [[NSWorkspace sharedWorkspace] accessibilityDisplayShouldReduceMotion];
#endif
}

#if PLATFORM(MAC)

ExceptionOr<RefPtr<Range>> Internals::rangeForDictionaryLookupAtLocation(int x, int y)
{
    auto* document = contextDocument();
    if (!document || !document->frame())
        return Exception { ExceptionCode::InvalidAccessError };

    document->updateLayout(LayoutOptions::IgnorePendingStylesheets);

    constexpr OptionSet<HitTestRequest::Type> hitType { HitTestRequest::Type::ReadOnly, HitTestRequest::Type::Active, HitTestRequest::Type::DisallowUserAgentShadowContent, HitTestRequest::Type::AllowChildFrameContent };
    
    auto* localFrame = dynamicDowncast<LocalFrame>(document->frame()->mainFrame());
    if (!localFrame)
        return nullptr; 

    auto result = localFrame->eventHandler().hitTestResultAtPoint(IntPoint(x, y), hitType);
    auto range = DictionaryLookup::rangeAtHitTestResult(result);
    if (!range)
        return nullptr;

    return RefPtr<Range> { createLiveRange(*range) };
}

void Internals::setUsesOverlayScrollbars(bool enabled)
{
    WebCore::DeprecatedGlobalSettings::setUsesOverlayScrollbars(enabled);

    ScrollerStyle::setUseOverlayScrollbars(enabled);

    ScrollbarTheme& theme = ScrollbarTheme::theme();
    if (theme.isMockTheme())
        return;

    static_cast<ScrollbarThemeMac&>(theme).preferencesChanged();

    NSScrollerStyle style = enabled ? NSScrollerStyleOverlay : NSScrollerStyleLegacy;
    [NSScrollerImpPair _updateAllScrollerImpPairsForNewRecommendedScrollerStyle:style];

    auto* document = contextDocument();
    if (!document || !document->frame())
        return;

    auto* localFrame = dynamicDowncast<LocalFrame>(document->frame()->mainFrame());
    if (!localFrame)
        return;

    localFrame->view()->scrollbarStyleDidChange();
}

#endif

#if ENABLE(VIDEO)
double Internals::privatePlayerVolume(const HTMLMediaElement& element)
{
    RefPtr corePlayer = element.player();
    if (!corePlayer)
        return 0;
    auto player = corePlayer->objCAVFoundationAVPlayer();
    if (!player)
        return 0;
    return [player volume];
}

bool Internals::privatePlayerMuted(const HTMLMediaElement& element)
{
    RefPtr corePlayer = element.player();
    if (!corePlayer)
        return false;
    auto player = corePlayer->objCAVFoundationAVPlayer();
    if (!player)
        return false;
    return [player isMuted];
}
#endif

String Internals::encodedPreferenceValue(const String& domain, const String& key)
{
    auto userDefaults = adoptNS([[NSUserDefaults alloc] initWithSuiteName:domain]);
    id value = [userDefaults objectForKey:key];
    auto data = retainPtr([NSKeyedArchiver archivedDataWithRootObject:value requiringSecureCoding:YES error:nullptr]);
    return [data base64EncodedStringWithOptions:0];
}

bool Internals::isRemoteUIAppForAccessibility()
{
#if PLATFORM(MAC)
    return [NSAccessibilityRemoteUIElement isRemoteUIApp];
#else
    return false;
#endif
}

bool Internals::hasSandboxIOKitOpenAccessToClass(const String& process, const String& ioKitClass)
{
    UNUSED_PARAM(process); // TODO: add support for getting PID of other WebKit processes.
    pid_t pid = getpid();

    return !sandbox_check(pid, "iokit-open", static_cast<enum sandbox_filter_type>(SANDBOX_FILTER_IOKIT_CONNECTION | SANDBOX_CHECK_NO_REPORT), ioKitClass.utf8().data());
}

#if ENABLE(DATA_DETECTION)

DDScannerResult *Internals::fakeDataDetectorResultForTesting()
{
    static NeverDestroyed result = []() -> RetainPtr<DDScannerResult> {
        auto scanner = adoptCF(PAL::softLink_DataDetectorsCore_DDScannerCreate(DDScannerTypeStandard, 0, nullptr));
        auto stringToScan = CFSTR("webkit.org");
        auto query = adoptCF(PAL::softLink_DataDetectorsCore_DDScanQueryCreateFromString(kCFAllocatorDefault, stringToScan, CFRangeMake(0, CFStringGetLength(stringToScan))));
        if (!PAL::softLink_DataDetectorsCore_DDScannerScanQuery(scanner.get(), query.get()))
            return nil;

        auto results = adoptCF(PAL::softLink_DataDetectorsCore_DDScannerCopyResultsWithOptions(scanner.get(), DDScannerCopyResultsOptionsNoOverlap));
        if (!CFArrayGetCount(results.get()))
            return nil;

        return { [[PAL::getDDScannerResultClass() resultsFromCoreResults:results.get()] firstObject] };
    }();
    return result->get();
}

#endif // ENABLE(DATA_DETECTION)

RefPtr<SharedBuffer> Internals::pngDataForTesting()
{
    NSBundle *webCoreBundle = [NSBundle bundleForClass:NSClassFromString(@"WebCoreBundleFinder")];
    return SharedBuffer::createWithContentsOfFile([webCoreBundle pathForResource:@"missingImage" ofType:@"png"]);
}

#if ENABLE(IMAGE_ANALYSIS_ENHANCEMENTS)

RetainPtr<VKCImageAnalysis> Internals::fakeImageAnalysisResultForTesting(const Vector<ImageOverlayLine>& lines)
{
    if (lines.isEmpty())
        return { };

    StringBuilder fullText;
    for (auto& line : lines) {
        for (auto& text : line.children) {
            if (text.hasLeadingWhitespace)
                fullText.append(space);
            fullText.append(text.text);
        }
        if (line.hasTrailingNewline)
            fullText.append(newlineCharacter);
    }

    return adoptNS((id)[[FakeImageAnalysisResult alloc] initWithString:fullText.toString()]);
}

#endif // ENABLE(IMAGE_ANALYSIS_ENHANCEMENTS)

} // namespace WebCore
