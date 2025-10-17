/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 7, 2023.
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

#import "AXObjectCache.h"
#import "AccessibilityObject.h"
#import "WebAccessibilityObjectWrapperBase.h"
#import "WAKView.h"

namespace WebCore {
class VisiblePosition;
}

// NSAttributedStrings support.

static NSString * const UIAccessibilityTextAttributeContext = @"UIAccessibilityTextAttributeContext";
static NSString * const UIAccessibilityTextualContextSourceCode = @"UIAccessibilityTextualContextSourceCode";

@interface WAKView (iOSAccessibility)
- (BOOL)accessibilityIsIgnored;
@end

@interface WebAccessibilityObjectWrapper : WebAccessibilityObjectWrapperBase {
    // Cached data to avoid frequent re-computation.
    int m_isAccessibilityElement;
    uint64_t m_accessibilityTraitsFromAncestor;
}

- (WebCore::AccessibilityObject *)axBackingObject;

- (id)accessibilityHitTest:(CGPoint)point;
- (AccessibilityObjectWrapper *)accessibilityPostProcessHitTest:(CGPoint)point;
- (BOOL)accessibilityCanFuzzyHitTest;

- (BOOL)isAccessibilityElement;
- (NSString *)accessibilityLabel;
- (CGRect)accessibilityFrame;
- (NSString *)accessibilityValue;

- (NSInteger)accessibilityElementCount;
- (id)accessibilityElementAtIndex:(NSInteger)index;
- (NSInteger)indexOfAccessibilityElement:(id)element;

- (BOOL)isAttachment;

// This interacts with Accessibility system to post-process some notifications.
- (void)accessibilityOverrideProcessNotification:(NSString *)notificationName notificationData:(NSData *)notificationData;

// This is called by the Accessibility system to relay back to the chrome.
- (void)handleNotificationRelayToChrome:(NSString *)notificationName notificationData:(NSData *)notificationData;

@end

@interface WebAccessibilityTextMarker : NSObject {
    WebCore::AXObjectCache* _cache;
    WebCore::TextMarkerData _textMarkerData;
}

+ (WebAccessibilityTextMarker *)textMarkerWithVisiblePosition:(WebCore::VisiblePosition&)visiblePos cache:(WebCore::AXObjectCache*)cache;
+ (WebAccessibilityTextMarker *)textMarkerWithCharacterOffset:(WebCore::CharacterOffset&)characterOffset cache:(WebCore::AXObjectCache*)cache;
+ (WebAccessibilityTextMarker *)startOrEndTextMarkerForRange:(const std::optional<WebCore::SimpleRange>&)range isStart:(BOOL)isStart cache:(WebCore::AXObjectCache*)cache;

- (id)initWithTextMarker:(const WebCore::TextMarkerData *)data cache:(WebCore::AXObjectCache*)cache;
- (WebCore::TextMarkerData)textMarkerData;
@end

#endif // PLATFORM(IOS_FAMILY)
