/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 20, 2023.
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
#import "AXCoreObject.h"
#import "FontPlatformData.h"
#import <CoreGraphics/CoreGraphics.h>
#import <variant>
#import <wtf/Markable.h>
#import <wtf/RefPtr.h>
#import <wtf/WeakPtr.h>

namespace WebCore {
struct AccessibilitySearchCriteria;
class AccessibilityObject;
class AXIsolatedObject;
class Document;
class IntRect;
class FloatPoint;
class HTMLTextFormControlElement;
class Path;
class VisiblePosition;

// NSAttributedString support.
// FIXME: move to a new AXAttributedStringBuilder class. For now, these
// functions are implemented in AccessibilityObjectCocoa.mm. Additional helper
// functions are implemented in AccessibilityObjectMac or IOS .mm respectively.
bool attributedStringContainsRange(NSAttributedString *, const NSRange&);
#if PLATFORM(MAC)
void attributedStringSetColor(NSMutableAttributedString *attrString, NSString *attribute, NSColor *, const NSRange&);
#endif // PLATFORM(MAC)
void attributedStringSetNumber(NSMutableAttributedString *, NSString *, NSNumber *, const NSRange&);
void attributedStringSetFont(NSMutableAttributedString *, CTFontRef, const NSRange&);
void attributedStringSetSpelling(NSMutableAttributedString *, Node&, StringView, const NSRange&);
RetainPtr<NSAttributedString> attributedStringCreate(Node&, StringView, const SimpleRange&, AXCoreObject::SpellCheck);
}

@interface WebAccessibilityObjectWrapperBase : NSObject {
    WeakPtr<WebCore::AccessibilityObject> m_axObject;

#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
    ThreadSafeWeakPtr<WebCore::AXIsolatedObject> m_isolatedObject;
    // To be accessed only on the main thread.
    bool m_isolatedObjectInitialized;
#endif

    Markable<WebCore::AXID> _identifier;
}

- (id)initWithAccessibilityObject:(WebCore::AccessibilityObject&)axObject;
#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
- (void)attachIsolatedObject:(WebCore::AXIsolatedObject*)isolatedObject;
- (BOOL)hasIsolatedObject;
#endif

- (void)detach;
#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
- (void)detachIsolatedObject:(WebCore::AccessibilityDetachmentType)detachmentType;
#endif

@property (nonatomic, assign) Markable<WebCore::AXID> identifier;

// FIXME: unified these two methods into one.
#if PLATFORM(MAC)
// Updates the underlying object and accessibility hierarchy , and returns the
// corresponding AXCoreObject.
- (WebCore::AXCoreObject*)updateObjectBackingStore;
#else
- (BOOL)_prepareAccessibilityCall;
#endif

// This can be either an AccessibilityObject or an AXIsolatedObject
- (WebCore::AXCoreObject*)axBackingObject;

- (NSArray<NSDictionary *> *)lineRectsAndText;

// These are pre-fixed with base so that AppKit does not end up calling into these directly (bypassing safety checks).
- (NSString *)baseAccessibilityHelpText;
- (NSArray<NSString *> *)baseAccessibilitySpeechHint;

- (NSString *)ariaLandmarkRoleDescription;

- (id)attachmentView;
// Used to inform an element when a notification is posted for it. Used by tests.
- (void)accessibilityPostedNotification:(NSString *)notificationName;
- (void)accessibilityPostedNotification:(NSString *)notificationName userInfo:(NSDictionary *)userInfo;

- (CGPathRef)convertPathToScreenSpace:(const WebCore::Path&)path;

- (CGRect)convertRectToSpace:(const WebCore::FloatRect&)rect space:(WebCore::AccessibilityConversionSpace)space;

// Math related functions
- (NSArray *)accessibilityMathPostscriptPairs;
- (NSArray *)accessibilityMathPrescriptPairs;

- (NSRange)accessibilityVisibleCharacterRange;

- (NSDictionary<NSString *, id> *)baseAccessibilityResolvedEditingStyles;

extern WebCore::AccessibilitySearchCriteria accessibilitySearchCriteriaForSearchPredicate(WebCore::AXCoreObject&, const NSDictionary *);

extern NSArray *makeNSArray(const WebCore::AXCoreObject::AccessibilityChildrenVector&, BOOL returnPlatformElements = YES);
extern NSRange makeNSRange(std::optional<WebCore::SimpleRange>);
extern std::optional<WebCore::SimpleRange> makeDOMRange(WebCore::Document*, NSRange);

#if PLATFORM(IOS_FAMILY)
- (id)_accessibilityWebDocumentView;
#endif

@end
