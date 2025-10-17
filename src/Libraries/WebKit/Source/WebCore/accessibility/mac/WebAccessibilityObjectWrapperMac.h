/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 2, 2024.
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
#import "WebAccessibilityObjectWrapperBase.h"

#if PLATFORM(MAC)

#import <pal/spi/mac/HIServicesSPI.h>
#import <wtf/RetainPtr.h>

#ifndef NSAccessibilityPrimaryScreenHeightAttribute
#define NSAccessibilityPrimaryScreenHeightAttribute @"_AXPrimaryScreenHeight"
#endif

#ifndef NSAccessibilityChildrenInNavigationOrderAttribute
#define NSAccessibilityChildrenInNavigationOrderAttribute @"AXChildrenInNavigationOrder"
#endif

// NSAttributedString support.

#ifndef NSAccessibilityBlockQuoteLevelAttribute
#define NSAccessibilityBlockQuoteLevelAttribute @"AXBlockQuoteLevel"
#endif

#ifndef NSAccessibilityExpandedTextValueAttribute
#define NSAccessibilityExpandedTextValueAttribute @"AXExpandedTextValue"
#endif

#ifndef NSAccessibilityTextCompletionAttribute
#define NSAccessibilityTextCompletionAttribute @"AXTextCompletion"
#endif

extern "C" AXUIElementRef NSAccessibilityCreateAXUIElementRef(id element);

// TextMarker and TextMarkerRange API constants.

// TextMarker attributes:
#define AXTextMarkerIsValidAttribute @"AXTextMarkerIsValid"
#define AXIndexForTextMarkerAttribute @"AXIndexForTextMarker"
#define AXTextMarkerForIndexAttribute @"AXTextMarkerForIndex"
#define AXNextTextMarkerForTextMarkerAttribute @"AXNextTextMarkerForTextMarker"
#define AXPreviousTextMarkerForTextMarkerAttribute @"AXPreviousTextMarkerForTextMarker"

// AXUIElement attributes:
#define AXUIElementForTextMarkerAttribute @"AXUIElementForTextMarker"
#define AXTextMarkerRangeForUIElementAttribute @"AXTextMarkerRangeForUIElement"
#define AXStartTextMarkerAttribute @"AXStartTextMarker"
#define AXEndTextMarkerAttribute @"AXEndTextMarker"

// TextMarkerRange creation:
#define AXTextMarkerRangeForTextMarkersAttribute @"AXTextMarkerRangeForTextMarkers"
#define AXTextMarkerRangeForUnorderedTextMarkersAttribute @"AXTextMarkerRangeForUnorderedTextMarkers"

// TextMarkerRange attributes:
#define AXLengthForTextMarkerRangeAttribute @"AXLengthForTextMarkerRange"

// Text extraction:
#define AXStringForTextMarkerRangeAttribute @"AXStringForTextMarkerRange"
#define AXAttributedStringForTextMarkerRangeAttribute @"AXAttributedStringForTextMarkerRange"
#define AXAttributedStringForTextMarkerRangeWithOptionsAttribute @"AXAttributedStringForTextMarkerRangeWithOptions"

// Geometry attributes:
#define AXTextMarkerForPositionAttribute @"AXTextMarkerForPosition" // FIXME: should be AXTextMarkerForPoint.
#define AXBoundsForTextMarkerRangeAttribute @"AXBoundsForTextMarkerRange"
#define AXStartTextMarkerForBoundsAttribute @"AXStartTextMarkerForBounds"
#define AXEndTextMarkerForBoundsAttribute @"AXEndTextMarkerForBounds"

// Line attributes:
#define AXLineForTextMarkerAttribute @"AXLineForTextMarker"
#define AXTextMarkerRangeForLineAttribute @"AXTextMarkerRangeForLine"
#define AXLineTextMarkerRangeForTextMarkerAttribute @"AXLineTextMarkerRangeForTextMarker"
#define AXLeftLineTextMarkerRangeForTextMarkerAttribute @"AXLeftLineTextMarkerRangeForTextMarker"
#define AXRightLineTextMarkerRangeForTextMarkerAttribute @"AXRightLineTextMarkerRangeForTextMarker"
#define AXNextLineEndTextMarkerForTextMarkerAttribute @"AXNextLineEndTextMarkerForTextMarker"
#define AXPreviousLineStartTextMarkerForTextMarkerAttribute @"AXPreviousLineStartTextMarkerForTextMarker"

// Word attributes:
#define AXLeftWordTextMarkerRangeForTextMarkerAttribute @"AXLeftWordTextMarkerRangeForTextMarker"
#define AXRightWordTextMarkerRangeForTextMarkerAttribute @"AXRightWordTextMarkerRangeForTextMarker"
#define AXNextWordEndTextMarkerForTextMarkerAttribute @"AXNextWordEndTextMarkerForTextMarker"
#define AXPreviousWordStartTextMarkerForTextMarkerAttribute @"AXPreviousWordStartTextMarkerForTextMarker"

// Sentence attributes:
#define AXSentenceTextMarkerRangeForTextMarkerAttribute @"AXSentenceTextMarkerRangeForTextMarker"
#define AXNextSentenceEndTextMarkerForTextMarkerAttribute @"AXNextSentenceEndTextMarkerForTextMarker"
#define AXPreviousSentenceStartTextMarkerForTextMarkerAttribute @"AXPreviousSentenceStartTextMarkerForTextMarker"

// Paragraph attributes:
#define AXParagraphTextMarkerRangeForTextMarkerAttribute @"AXParagraphTextMarkerRangeForTextMarker"
#define AXNextParagraphEndTextMarkerForTextMarkerAttribute @"AXNextParagraphEndTextMarkerForTextMarker"
#define AXPreviousParagraphStartTextMarkerForTextMarkerAttribute @"AXPreviousParagraphStartTextMarkerForTextMarker"

// Other ranges:
#define AXDidSpellCheckAttribute @"AXDidSpellCheck"
#define AXMisspellingTextMarkerRangeAttribute @"AXMisspellingTextMarkerRange"
#define AXSelectedTextMarkerRangeAttribute @"AXSelectedTextMarkerRange"
#define AXStyleTextMarkerRangeForTextMarkerAttribute @"AXStyleTextMarkerRangeForTextMarker"

// Private attributes exposed only for testing:
#define _AXStartTextMarkerForTextMarkerRangeAttribute @"_AXStartTextMarkerForTextMarkerRange"
#define _AXEndTextMarkerForTextMarkerRangeAttribute @"_AXEndTextMarkerForTextMarkerRange"
#define _AXTextMarkerRangeForNSRangeAttribute @"_AXTextMarkerRangeForNSRange"

#if ENABLE(TREE_DEBUGGING)
#define AXTextMarkerDebugDescriptionAttribute @"AXTextMarkerDebugDescription"
#define AXTextMarkerRangeDebugDescriptionAttribute @"AXTextMarkerRangeDebugDescription"
#define AXTextMarkerNodeDebugDescriptionAttribute @"AXTextMarkerNodeDebugDescription"
#define AXTextMarkerNodeTreeDebugDescriptionAttribute @"AXTextMarkerNodeTreeDebugDescription"
#endif

@interface WebAccessibilityObjectWrapper : WebAccessibilityObjectWrapperBase

// When a plugin uses a WebKit control to act as a surrogate view (e.g. PDF use WebKit to create text fields).
- (id)_associatedPluginParent;
// For testing use only.
- (void)_accessibilityHitTestResolvingRemoteFrame:(NSPoint)point callback:(void(^)(NSString *))callback;
- (NSArray *)_accessibilityChildrenFromIndex:(NSUInteger)index maxCount:(NSUInteger)maxCount returnPlatformElements:(BOOL)returnPlatformElements;

@end

#else

typedef const struct __AXTextMarker* AXTextMarkerRef;
typedef const struct __AXTextMarkerRange* AXTextMarkerRangeRef;

#endif // PLATFORM(MAC)

namespace WebCore {

class AccessibilityObject;
struct CharacterOffset;

// TextMarker and TextMarkerRange public funcstions.
// FIXME: TextMarker and TextMarkerRange should become classes on their own right, wrapping the system objects.

RetainPtr<AXTextMarkerRangeRef> textMarkerRangeFromMarkers(AXTextMarkerRef, AXTextMarkerRef);
AccessibilityObject* accessibilityObjectForTextMarker(AXObjectCache*, AXTextMarkerRef);

// TextMarker <-> VisiblePosition conversion.
AXTextMarkerRef textMarkerForVisiblePosition(AXObjectCache*, const VisiblePosition&);
VisiblePosition visiblePositionForTextMarker(AXObjectCache*, AXTextMarkerRef);

// TextMarkerRange <-> VisiblePositionRange conversion.
AXTextMarkerRangeRef textMarkerRangeFromVisiblePositions(AXObjectCache*, const VisiblePosition&, const VisiblePosition&);
VisiblePositionRange visiblePositionRangeForTextMarkerRange(AXObjectCache*, AXTextMarkerRangeRef);

// TextMarker <-> CharacterOffset conversion.
AXTextMarkerRef textMarkerForCharacterOffset(AXObjectCache*, const CharacterOffset&);
CharacterOffset characterOffsetForTextMarker(AXObjectCache*, AXTextMarkerRef);

// TextMarkerRange <-> SimpleRange conversion.
AXTextMarkerRef startOrEndTextMarkerForRange(AXObjectCache*, const std::optional<SimpleRange>&, bool isStart);
AXTextMarkerRangeRef textMarkerRangeFromRange(AXObjectCache*, const std::optional<SimpleRange>&);
std::optional<SimpleRange> rangeForTextMarkerRange(AXObjectCache*, AXTextMarkerRangeRef);

NSArray *renderWidgetChildren(const AXCoreObject&);

} // namespace WebCore
