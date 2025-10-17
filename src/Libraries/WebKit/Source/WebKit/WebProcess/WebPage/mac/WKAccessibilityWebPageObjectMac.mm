/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 2, 2022.
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
#import "WKAccessibilityWebPageObjectMac.h"

#if PLATFORM(MAC)

#import "PluginView.h"
#import "WebFrame.h"
#import "WebPage.h"
#import "WKArray.h"
#import "WKNumber.h"
#import "WKRetainPtr.h"
#import "WKSharedAPICast.h"
#import "WKString.h"
#import "WKStringCF.h"
#import <WebCore/AXObjectCache.h>
#import <WebCore/LocalFrame.h>
#import <WebCore/LocalFrameView.h>
#import <WebCore/Page.h>
#import <WebCore/PageOverlayController.h>
#import <WebCore/PlatformScreen.h>
#import <WebCore/ScrollView.h>
#import <WebCore/Scrollbar.h>
#import <WebCore/WebAccessibilityObjectWrapperMac.h>
#import <pal/spi/cocoa/NSAccessibilitySPI.h>
#import <wtf/cocoa/VectorCocoa.h>

namespace ax = WebCore::Accessibility;

@implementation WKAccessibilityWebPageObject

#define PROTECTED_SELF protectedSelf = RetainPtr<WKAccessibilityWebPageObject>(self)

- (instancetype)init
{
    self = [super init];
    if (!self)
        return self;

    self->m_attributeNames = adoptNS([[NSArray alloc] initWithObjects:
        NSAccessibilityRoleAttribute, NSAccessibilityRoleDescriptionAttribute, NSAccessibilityFocusedAttribute,
        NSAccessibilityParentAttribute, NSAccessibilityWindowAttribute, NSAccessibilityTopLevelUIElementAttribute,
        NSAccessibilityPositionAttribute, NSAccessibilitySizeAttribute, NSAccessibilityChildrenAttribute, NSAccessibilityChildrenInNavigationOrderAttribute, NSAccessibilityPrimaryScreenHeightAttribute, nil]);
    return self;
}

- (void)dealloc
{
    NSAccessibilityUnregisterUniqueIdForUIElement(self);
    [super dealloc];
}

- (void)setWebPage:(NakedPtr<WebKit::WebPage>)page
{
    ASSERT(isMainRunLoop());
    [super setWebPage:page];

    if (!page) {
        m_parameterizedAttributeNames = @[];
        return;
    }

    auto* corePage = page->corePage();
    if (!corePage) {
        m_parameterizedAttributeNames = @[];
        return;
    }

    m_parameterizedAttributeNames = createNSArray(corePage->pageOverlayController().copyAccessibilityAttributesNames(true));
    // FIXME: m_parameterizedAttributeNames needs to be updated when page overlays are added or removed, although this is a property that doesn't change much.
}

ALLOW_DEPRECATED_IMPLEMENTATIONS_BEGIN
- (BOOL)accessibilityIsIgnored
ALLOW_DEPRECATED_IMPLEMENTATIONS_END
{
    return NO;
}

ALLOW_DEPRECATED_IMPLEMENTATIONS_BEGIN
- (NSArray *)accessibilityAttributeNames
ALLOW_DEPRECATED_IMPLEMENTATIONS_END
{
    return m_attributeNames.get();
}

ALLOW_DEPRECATED_IMPLEMENTATIONS_BEGIN
- (NSArray *)accessibilityParameterizedAttributeNames
ALLOW_DEPRECATED_IMPLEMENTATIONS_END
{
    return m_parameterizedAttributeNames.get();
}

ALLOW_DEPRECATED_IMPLEMENTATIONS_BEGIN
- (BOOL)accessibilityIsAttributeSettable:(NSString *)attribute
ALLOW_DEPRECATED_IMPLEMENTATIONS_END
{
    return NO;
}

ALLOW_DEPRECATED_IMPLEMENTATIONS_BEGIN
- (void)accessibilitySetValue:(id)value forAttribute:(NSString *)attribute
ALLOW_DEPRECATED_IMPLEMENTATIONS_END
{
}

- (NSPoint)convertScreenPointToRootView:(NSPoint)point
{
    return ax::retrieveValueFromMainThread<NSPoint>([&point, PROTECTED_SELF] () -> NSPoint {
        if (!protectedSelf->m_page)
            return point;
        return protectedSelf->m_page->screenToRootView(WebCore::IntPoint(point.x, point.y));
    });
}

ALLOW_DEPRECATED_IMPLEMENTATIONS_BEGIN
- (NSArray *)accessibilityActionNames
ALLOW_DEPRECATED_IMPLEMENTATIONS_END
{
    return @[];
}

- (NSArray *)accessibilityChildren
{
    id wrapper = [self accessibilityRootObjectWrapper];
    return wrapper ? @[wrapper] : @[];
}

- (NSArray *)accessibilityChildrenInNavigationOrder
{
    return [self accessibilityChildren];
}

ALLOW_DEPRECATED_IMPLEMENTATIONS_BEGIN
- (id)accessibilityAttributeValue:(NSString *)attribute
ALLOW_DEPRECATED_IMPLEMENTATIONS_END
{
    static std::atomic<bool> didInitialize { false };
    static std::atomic<unsigned> screenHeight { 0 };
    if (UNLIKELY(!didInitialize)) {
        didInitialize = true;
        callOnMainRunLoopAndWait([protectedSelf = retainPtr(self)] {
            if (!WebCore::AXObjectCache::accessibilityEnabled())
                [protectedSelf enableAccessibilityForAllProcesses];

#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
            if (WebCore::AXObjectCache::isIsolatedTreeEnabled())
                WebCore::AXObjectCache::initializeAXThreadIfNeeded();
#endif // ENABLE(ACCESSIBILITY_ISOLATED_TREE)

            float roundedHeight = std::round(WebCore::screenRectForPrimaryScreen().size().height());
            screenHeight = std::max(0u, static_cast<unsigned>(roundedHeight));
        });
    }

    // The following attributes can be handled off the main thread.

    if ([attribute isEqualToString:NSAccessibilityRoleAttribute])
        return NSAccessibilityGroupRole;

    if ([attribute isEqualToString:NSAccessibilityRoleDescriptionAttribute])
        return NSAccessibilityRoleDescription(NSAccessibilityGroupRole, nil);

    if ([attribute isEqualToString:NSAccessibilityFocusedAttribute])
        return @NO;

    if ([attribute isEqualToString:NSAccessibilityPositionAttribute])
        return [self accessibilityAttributePositionValue];

    if ([attribute isEqualToString:NSAccessibilitySizeAttribute])
        return [self accessibilityAttributeSizeValue];

    if ([attribute isEqualToString:NSAccessibilityChildrenAttribute]
        || [attribute isEqualToString:NSAccessibilityChildrenInNavigationOrderAttribute]) {
        // The root object is the only child.
        return [self accessibilityChildren];
    }

    if ([attribute isEqualToString:NSAccessibilityParentAttribute])
        return [self accessibilityAttributeParentValue].get();

    if ([attribute isEqualToString:NSAccessibilityPrimaryScreenHeightAttribute])
        return @(screenHeight.load());

    if ([attribute isEqualToString:NSAccessibilityWindowAttribute])
        return [self accessibilityAttributeWindowValue].get();

    if ([attribute isEqualToString:NSAccessibilityTopLevelUIElementAttribute])
        return [self accessibilityAttributeTopLevelUIElementValue].get();

    return nil;
}

- (NSValue *)accessibilityAttributeSizeValue
{
#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
    if (!isMainRunLoop()) {
        Locker lock { m_cacheLock };
        return [NSValue valueWithSize:(NSSize)m_size];
    }
#endif // ENABLE(ACCESSIBILITY_ISOLATED_TREE)

    return m_page ? [NSValue valueWithSize:m_page->size()] : nil;
}

- (NSValue *)accessibilityAttributePositionValue
{
#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
    if (!isMainRunLoop()) {
        Locker lock { m_cacheLock };
        return [NSValue valueWithPoint:(NSPoint)m_position];
    }
#endif // ENABLE(ACCESSIBILITY_ISOLATED_TREE)

    return m_page ? [NSValue valueWithPoint:m_page->accessibilityPosition()] : nil;
}

- (RetainPtr<id>)accessibilityAttributeParentValue
{
#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
    if (!isMainRunLoop()) {
        Locker lock { m_parentLock };
        return m_parent;
    }
#endif // ENABLE(ACCESSIBILITY_ISOLATED_TREE)

    return m_parent;
}

// FIXME: accessibilityAttributeWindowValue and accessibilityAttributeTopLevelUIElementValue
// always return nil for instances of this class when set up by WebPage::registerRemoteFrameAccessibilityTokens,
// as nothing there sets m_window, setWindowUIElement, and setTopLevelUIElement.
- (RetainPtr<id>)accessibilityAttributeWindowValue
{
#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
    if (!isMainRunLoop()) {
        // Use the cached window to avoid using m_parent (which is possibly an AppKit object) off the main-thread.
        Locker lock { m_windowLock };
        return m_window.get();
    }
#endif // ENABLE(ACCESSIBILITY_ISOLATED_TREE)

    ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    return [m_parent accessibilityAttributeValue:NSAccessibilityWindowAttribute];
    ALLOW_DEPRECATED_DECLARATIONS_END
}

- (RetainPtr<id>)accessibilityAttributeTopLevelUIElementValue
{
#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
    if (!isMainRunLoop()) {
        // Use the cached window to avoid using m_parent (which is possibly an AppKit object) off the main-thread.
        // The TopLevelUIElement is the window, as we set it as such in WebPage::registerUIProcessAccessibilityTokens,
        // so we can return m_window here.
        Locker lock { m_windowLock };
        return m_window.get();
    }
#endif // ENABLE(ACCESSIBILITY_ISOLATED_TREE)

    ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    return [m_parent accessibilityAttributeValue:NSAccessibilityTopLevelUIElementAttribute];
    ALLOW_DEPRECATED_DECLARATIONS_END
}

- (id)accessibilityDataDetectorValue:(NSString *)attribute point:(WebCore::FloatPoint&)point
{
    return ax::retrieveValueFromMainThread<RetainPtr<id>>([&attribute, &point, PROTECTED_SELF] () -> RetainPtr<id> {
        if (!protectedSelf->m_page)
            return nil;
        id value = nil;
        if ([attribute isEqualToString:@"AXDataDetectorExistsAtPoint"] || [attribute isEqualToString:@"AXDidShowDataDetectorMenuAtPoint"]) {
            bool boolValue;
            if (protectedSelf->m_page->corePage()->pageOverlayController().copyAccessibilityAttributeBoolValueForPoint(attribute, point, boolValue))
                value = [NSNumber numberWithBool:boolValue];
        }
        if ([attribute isEqualToString:@"AXDataDetectorTypeAtPoint"]) {
            String stringValue;
            if (protectedSelf->m_page->corePage()->pageOverlayController().copyAccessibilityAttributeStringValueForPoint(attribute, point, stringValue))
                value = [NSString stringWithString:stringValue];
        }
        return value;
    }).autorelease();
}

ALLOW_DEPRECATED_IMPLEMENTATIONS_BEGIN
- (id)accessibilityAttributeValue:(NSString *)attribute forParameter:(id)parameter
ALLOW_DEPRECATED_IMPLEMENTATIONS_END
{
    WebCore::FloatPoint pageOverlayPoint;
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    if ([parameter isKindOfClass:[NSValue class]] && !strcmp([(NSValue *)parameter objCType], @encode(NSPoint)))
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
        pageOverlayPoint = [self convertScreenPointToRootView:[(NSValue *)parameter pointValue]];
    else
        return nil;

    if ([attribute isEqualToString:@"AXDataDetectorExistsAtPoint"] || [attribute isEqualToString:@"AXDidShowDataDetectorMenuAtPoint"] || [attribute isEqualToString:@"AXDataDetectorTypeAtPoint"])
        return [self accessibilityDataDetectorValue:attribute point:pageOverlayPoint];

    return nil;
}

- (BOOL)accessibilityShouldUseUniqueId
{
    return YES;
}

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
- (id)accessibilityHitTest:(NSPoint)point
{
    auto convertedPoint = ax::retrieveValueFromMainThread<WebCore::IntPoint>([&point, PROTECTED_SELF] () -> WebCore::IntPoint {
        if (!protectedSelf->m_page)
            return WebCore::IntPoint(point);

        // PDF plug-in handles the scroll view offset natively as part of the layer conversions.
        if (protectedSelf->m_page->mainFramePlugIn())
            return WebCore::IntPoint(point);

        auto convertedPoint = protectedSelf->m_page->screenToRootView(WebCore::IntPoint(point));

        if (CheckedPtr localFrameView = protectedSelf->m_page->localMainFrameView())
            convertedPoint.moveBy(localFrameView->scrollPosition());
        else if (RefPtr remoteLocalFrame = [protectedSelf remoteLocalFrame]) {
            if (CheckedPtr frameView = remoteLocalFrame->view())
                convertedPoint.moveBy(frameView->scrollPosition());
        }
        if (auto* page = protectedSelf->m_page->corePage())
            convertedPoint.move(0, -page->topContentInset());
        return convertedPoint;
    });
    
    return [[self accessibilityRootObjectWrapper] accessibilityHitTest:convertedPoint];
}
ALLOW_DEPRECATED_DECLARATIONS_END

@end

#endif // PLATFORM(MAC)

