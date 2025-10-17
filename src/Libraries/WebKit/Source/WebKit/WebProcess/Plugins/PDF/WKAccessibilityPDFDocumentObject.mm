/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 23, 2023.
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
#include "config.h"
#include "WKAccessibilityPDFDocumentObject.h"

#if ENABLE(UNIFIED_PDF) && PLATFORM(MAC)

#include "PDFKitSPI.h"
#include "PDFPluginAnnotation.h"
#include "PDFPluginBase.h"
#include "UnifiedPDFPlugin.h"
#include <PDFKit/PDFKit.h>
#include <WebCore/AXObjectCache.h>
#include <WebCore/HTMLPlugInElement.h>
#include <WebCore/WebAccessibilityObjectWrapperMac.h>
#include <pal/spi/cocoa/NSAccessibilitySPI.h>
#include <wtf/CheckedPtr.h>
#include <wtf/RetainPtr.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/WeakObjCPtr.h>

@implementation WKAccessibilityPDFDocumentObject

@synthesize pluginElement = _pluginElement;

- (id)initWithPDFDocument:(RetainPtr<PDFDocument>)document andElement:(WebCore::HTMLPlugInElement*)element
{
    if (!(self = [super init]))
        return nil;

    _pdfDocument = document;
    _pluginElement = element;
    // We are setting the presenter ID of the WKAccessibilityPDFDocumentObject to the hosting application's PID.
    // This way VoiceOver can set AX observers on all the PDF AX nodes which are descendant of this element.
    if ([self respondsToSelector:@selector(accessibilitySetPresenterProcessIdentifier:)])
        [(id)self accessibilitySetPresenterProcessIdentifier:legacyPresentingApplicationPID()];
    return self;
}

- (void)setPDFPlugin:(WebKit::UnifiedPDFPlugin*)plugin
{
    _pdfPlugin = plugin;
}

- (void)setPDFDocument:(RetainPtr<PDFDocument>)document
{
    _pdfDocument = document;
}

- (BOOL)isAccessibilityElement
{
    return YES;
}

- (id)accessibilityFocusedUIElement
{
    if (RefPtr plugin = _pdfPlugin.get()) {
        if (RefPtr activeAnnotation = plugin->activeAnnotation()) {
            if (WebCore::AXObjectCache* existingCache = plugin->axObjectCache()) {
                if (RefPtr object = existingCache->getOrCreate(activeAnnotation->element())) {
                ALLOW_DEPRECATED_DECLARATIONS_BEGIN
                    return [object->wrapper() accessibilityAttributeValue:@"_AXAssociatedPluginParent"];
                ALLOW_DEPRECATED_DECLARATIONS_END
                }
            }
        }
    }
    for (id page in [self accessibilityChildren]) {
        id focusedElement = [page accessibilityFocusedUIElement];
        if (focusedElement)
            return focusedElement;
    }
    return nil;
}

- (id)accessibilityWindow
{
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    return [[self accessibilityParent]  accessibilityAttributeValue:NSAccessibilityWindowAttribute];
ALLOW_DEPRECATED_DECLARATIONS_END
}

- (id)accessibilityTopLevelUIElement
{
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    return [[self accessibilityParent] accessibilityAttributeValue:NSAccessibilityTopLevelUIElementAttribute];
ALLOW_DEPRECATED_DECLARATIONS_END
}

- (PDFDocument*)document
{
    return _pdfDocument.get();
}

- (NSArray *)accessibilityVisibleChildren
{
    RetainPtr<NSMutableArray> visiblePageElements = adoptNS([[NSMutableArray alloc] init]);
    for (id page in [self accessibilityChildren]) {
        id focusedElement = [page accessibilityFocusedUIElement];
        if (focusedElement)
            [visiblePageElements addObject:page];
    }
    return visiblePageElements.autorelease();
}

- (NSString *)accessibilitySubrole
{
    return @"AXPDFPluginSubrole";
}

- (NSRect)accessibilityFrame
{
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    id accessibilityParent = [self accessibilityParent];
    NSSize size = [[accessibilityParent accessibilityAttributeValue:NSAccessibilitySizeAttribute] sizeValue];
    NSPoint origin = [[accessibilityParent accessibilityAttributeValue:NSAccessibilityPositionAttribute] pointValue];
ALLOW_DEPRECATED_DECLARATIONS_END
    return NSMakeRect(origin.x, origin.y, size.width, size.height);
}

- (NSObject *)accessibilityParent
{
    RetainPtr protectedSelf = self;
    if (!protectedSelf->_parent) {
        callOnMainRunLoopAndWait([protectedSelf] {
            if (CheckedPtr axObjectCache = protectedSelf->_pdfPlugin.get()->axObjectCache()) {
                if (RefPtr pluginAxObject = axObjectCache->getOrCreate(protectedSelf->_pluginElement.get()))
                    protectedSelf->_parent = pluginAxObject->wrapper();
            }
        });
    }
    return protectedSelf->_parent.get().get();
}

- (void)setParent:(NSObject *)parent
{
    _parent = parent;
}

ALLOW_DEPRECATED_IMPLEMENTATIONS_BEGIN
- (id)accessibilityAttributeValue:(NSString *)attribute
{
    if ([attribute isEqualToString:NSAccessibilityParentAttribute])
        return [self accessibilityParent];
    if ([attribute isEqualToString:NSAccessibilityChildrenAttribute])
        return [self accessibilityChildren];
    if ([attribute isEqualToString:NSAccessibilityVisibleChildrenAttribute])
        return [self accessibilityVisibleChildren];
    if ([attribute isEqualToString:NSAccessibilityTopLevelUIElementAttribute])
        return [self accessibilityTopLevelUIElement];
    if ([attribute isEqualToString:NSAccessibilityWindowAttribute])
        return [self accessibilityWindow];
    if ([attribute isEqualToString:NSAccessibilityEnabledAttribute])
        return [[self accessibilityParent] accessibilityAttributeValue:NSAccessibilityEnabledAttribute];
    if ([attribute isEqualToString:NSAccessibilityRoleAttribute])
        return NSAccessibilityGroupRole;
    if ([attribute isEqualToString:NSAccessibilityPrimaryScreenHeightAttribute])
        return [[self accessibilityParent] accessibilityAttributeValue:NSAccessibilityPrimaryScreenHeightAttribute];
    if ([attribute isEqualToString:NSAccessibilitySubroleAttribute])
        return [self accessibilitySubrole];
    if ([attribute isEqualToString:NSAccessibilitySizeAttribute]) {
        if (RefPtr plugin = _pdfPlugin.get())
            return [NSValue valueWithSize:plugin->boundsOnScreen().size()];
    }
    if ([attribute isEqualToString:NSAccessibilityPositionAttribute]) {
        if (RefPtr plugin = _pdfPlugin.get())
            return [NSValue valueWithPoint:plugin->boundsOnScreen().location()];
    }
    return nil;
}
ALLOW_DEPRECATED_IMPLEMENTATIONS_END

ALLOW_DEPRECATED_IMPLEMENTATIONS_BEGIN
- (NSArray *)accessibilityAttributeNames
{
    static NeverDestroyed<RetainPtr<NSArray>> attributeNames = @[
        NSAccessibilityParentAttribute,
        NSAccessibilityWindowAttribute,
        NSAccessibilityTopLevelUIElementAttribute,
        NSAccessibilityRoleDescriptionAttribute,
        NSAccessibilitySizeAttribute,
        NSAccessibilityEnabledAttribute,
        NSAccessibilityPositionAttribute,
        NSAccessibilityFocusedAttribute,
        NSAccessibilityChildrenAttribute,
        NSAccessibilityPrimaryScreenHeightAttribute,
        NSAccessibilitySubroleAttribute
    ];
    return attributeNames.get().get();
}
ALLOW_DEPRECATED_IMPLEMENTATIONS_END

- (BOOL)accessibilityShouldUseUniqueId
{
    return YES;
}

- (NSUInteger)accessibilityArrayAttributeCount:(NSString *)attribute
{
    if (!_pdfDocument) {
        if (RefPtr plugin = _pdfPlugin.get())
            _pdfDocument = plugin->pdfDocument();
    }
    if ([attribute isEqualToString:NSAccessibilityChildrenAttribute])
        return [_pdfDocument.get() pageCount];
    if ([attribute isEqualToString:NSAccessibilityVisibleChildrenAttribute])
        return [self accessibilityVisibleChildren].count;
    return [super accessibilityArrayAttributeCount:attribute];
}

- (NSArray*)accessibilityChildren
{
    if (!_pdfDocument) {
        if (RefPtr plugin = _pdfPlugin.get())
            _pdfDocument = plugin->pdfDocument();
    }

    if ([_pdfDocument respondsToSelector:@selector(accessibilityChildren:)])
        return [_pdfDocument accessibilityChildren:self];

    return nil;
}

- (NSRect)convertFromPDFPageToScreenForAccessibility:(NSRect)rectInPageCoordinate pageIndex:(WebKit::PDFDocumentLayout::PageIndex)pageIndex
{
    if (RefPtr plugin = _pdfPlugin.get())
        return plugin->convertFromPDFPageToScreenForAccessibility(rectInPageCoordinate, pageIndex);
    return rectInPageCoordinate;
}

- (id)accessibilityAssociatedControlForAnnotation:(PDFAnnotation *)annotation
{
    id wrapper = nil;
    callOnMainRunLoopAndWait([protectedSelf = retainPtr(self), &wrapper] {
        RefPtr activeAnnotation = protectedSelf->_pdfPlugin.get()->activeAnnotation();
        if (!activeAnnotation)
            return;

        if (auto* axObjectCache = protectedSelf->_pdfPlugin.get()->axObjectCache()) {
            if (RefPtr annotationElementAxObject = axObjectCache->getOrCreate(activeAnnotation->element()))
                wrapper = annotationElementAxObject->wrapper();
        }
    });
    return wrapper;
}

- (void)setActiveAnnotation:(PDFAnnotation *)annotation
{
    RefPtr plugin = _pdfPlugin.get();
    plugin->setActiveAnnotation({ WTFMove(annotation) });
}

- (id)accessibilityHitTest:(NSPoint)point
{
    for (id element in [self accessibilityChildren]) {
        id result = [element accessibilityHitTest:point];
        if (result)
            return result;
    }
    return self;
}

// this function allows VoiceOver to scroll to the current page with VO cursor
- (void)gotoDestination:(PDFDestination *)destination
{
    RefPtr plugin = _pdfPlugin.get();
    if (!plugin)
        return;

    WebKit::PDFDocumentLayout::PageIndex pageIndex = [_pdfDocument indexForPage:[destination page]];

    callOnMainRunLoop([plugin, pageIndex] {
        plugin->accessibilityScrollToPage(pageIndex);
    });
}
@end

#endif
