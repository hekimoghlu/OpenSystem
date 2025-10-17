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
#import "config.h"
#import "AccessibilityObject.h"

#if PLATFORM(COCOA)

#import "AXObjectCache.h"
#import "TextIterator.h"
#import "WebAccessibilityObjectWrapperBase.h"
#import <wtf/cocoa/TypeCastsCocoa.h>

namespace WebCore {

String AccessibilityObject::speechHintAttributeValue() const
{
    auto speak = speakAsProperty();
    NSMutableArray<NSString *> *hints = [NSMutableArray array];
    [hints addObject:(speak & SpeakAs::SpellOut) ? @"spell-out" : @"normal"];
    if (speak & SpeakAs::Digits)
        [hints addObject:@"digits"];
    if (speak & SpeakAs::LiteralPunctuation)
        [hints addObject:@"literal-punctuation"];
    if (speak & SpeakAs::NoPunctuation)
        [hints addObject:@"no-punctuation"];
    return [hints componentsJoinedByString:@" "];
}

FloatPoint AccessibilityObject::screenRelativePosition() const
{
    auto rect = snappedIntRect(elementRect());
    // The Cocoa accessibility API wants the lower-left corner.
    return convertRectToPlatformSpace(FloatRect(FloatPoint(rect.x(), rect.maxY()), FloatSize()), AccessibilityConversionSpace::Screen).location();
}

AXTextMarkerRange AccessibilityObject::textMarkerRangeForNSRange(const NSRange& range) const
{
    if (range.location == NSNotFound)
        return { };

    if (!isTextControl())
        return { visiblePositionForIndex(range.location), visiblePositionForIndex(range.location + range.length) };

    if (range.location + range.length > text().length())
        return { };

    if (auto* cache = axObjectCache()) {
        auto start = cache->characterOffsetForIndex(range.location, this);
        auto end = cache->characterOffsetForIndex(range.location + range.length, this);
        return cache->rangeForUnorderedCharacterOffsets(start, end);
    }
    return { };
}

// NSAttributedString support.

#ifndef NSAttachmentCharacter
#define NSAttachmentCharacter 0xfffc
#endif

static void addObjectWrapperToArray(const AccessibilityObject& object, NSMutableArray *array)
{
    auto* wrapper = object.wrapper();
    if (!wrapper)
        return;

    // Don't add the same object twice.
    if ([array containsObject:wrapper])
        return;

#if PLATFORM(IOS_FAMILY)
    // Explicitly set that this is a new element, in case other logic tries to override.
    [wrapper setValue:@YES forKey:@"isAccessibilityElement"];
#endif

    [array addObject:wrapper];
}

void attributedStringSetNumber(NSMutableAttributedString *attrString, NSString *attribute, NSNumber *number, const NSRange& range)
{
    if (!attributedStringContainsRange(attrString, range))
        return;

    if (number)
        [attrString addAttribute:attribute value:number range:range];
}

static void attributedStringAppendWrapper(NSMutableAttributedString *attrString, WebAccessibilityObjectWrapper *wrapper)
{
    const auto attachmentCharacter = static_cast<UniChar>(NSAttachmentCharacter);
    [attrString appendAttributedString:adoptNS([[NSMutableAttributedString alloc] initWithString:[NSString stringWithCharacters:&attachmentCharacter length:1]
#if PLATFORM(MAC)
        attributes:@{ NSAccessibilityAttachmentTextAttribute : (__bridge id)adoptCF(NSAccessibilityCreateAXUIElementRef(wrapper)).get() }
#else
        attributes:@{ AccessibilityTokenAttachment : wrapper }
#endif
    ]).get()];
}

RetainPtr<NSArray> AccessibilityObject::contentForRange(const SimpleRange& range, SpellCheck spellCheck) const
{
    auto result = adoptNS([[NSMutableArray alloc] init]);

    // Iterate over the range to build the AX attributed strings.
    TextIterator it = textIteratorIgnoringFullSizeKana(range);
    for (; !it.atEnd(); it.advance()) {
        Node& node = it.range().start.container;

        // Non-zero length means textual node, zero length means replaced node (AKA "attachments" in AX).
        if (it.text().length()) {
            auto listMarkerText = listMarkerTextForNodeAndPosition(&node, makeContainerOffsetPosition(it.range().start));
            if (!listMarkerText.isEmpty()) {
                if (auto attrString = attributedStringCreate(node, listMarkerText, it.range(), SpellCheck::No))
                    [result addObject:attrString.get()];
            }

            if (auto attrString = attributedStringCreate(node, it.text(), it.range(), spellCheck))
                [result addObject:attrString.get()];
        } else {
            if (RefPtr replacedNode = it.node()) {
                auto* cache = axObjectCache();
                if (auto* object = cache ? cache->getOrCreate(replacedNode->renderer()) : nullptr)
                    addObjectWrapperToArray(*object, result.get());
            }
        }
    }

    return result;
}

RetainPtr<NSAttributedString> AccessibilityObject::attributedStringForTextMarkerRange(AXTextMarkerRange&& textMarkerRange, SpellCheck spellCheck) const
{
#if PLATFORM(MAC)
    auto range = rangeForTextMarkerRange(axObjectCache(), textMarkerRange);
#else
    auto range = textMarkerRange.simpleRange();
#endif
    return range ? attributedStringForRange(*range, spellCheck) : nil;
}

RetainPtr<NSAttributedString> AccessibilityObject::attributedStringForRange(const SimpleRange& range, SpellCheck spellCheck) const
{
    auto result = adoptNS([[NSMutableAttributedString alloc] init]);

    auto contents = contentForRange(range, spellCheck);
    for (id content in contents.get()) {
        auto item = retainPtr(content);
        if ([item isKindOfClass:[WebAccessibilityObjectWrapper class]]) {
            attributedStringAppendWrapper(result.get(), item.get());
            continue;
        }

        if (![item isKindOfClass:[NSAttributedString class]])
            continue;

        [result appendAttributedString:item.get()];
    }

    return result;
}

RetainPtr<CTFontRef> fontFrom(const RenderStyle& style)
{
    return style.fontCascade().primaryFont().getCTFont();
}

Color textColorFrom(const RenderStyle& style)
{
    return style.visitedDependentColor(CSSPropertyColor);
}

Color backgroundColorFrom(const RenderStyle& style)
{
    return style.visitedDependentColor(CSSPropertyBackgroundColor);
}

RetainPtr<CTFontRef> AccessibilityObject::font() const
{
    const auto* style = this->style();
    return style ? fontFrom(*style) : nil;
}

Color AccessibilityObject::textColor() const
{
    const auto* style = this->style();
    return style ? textColorFrom(*style) : Color();
}

Color AccessibilityObject::backgroundColor() const
{
    const auto* style = this->style();
    return style ? backgroundColorFrom(*style) : Color();
}

bool AccessibilityObject::isSubscript() const
{
    const auto* style = this->style();
    return style && style->verticalAlign() == VerticalAlign::Sub;
}

bool AccessibilityObject::isSuperscript() const
{
    const auto* style = this->style();
    return style && style->verticalAlign() == VerticalAlign::Super;
}

bool AccessibilityObject::hasTextShadow() const
{
    const auto* style = this->style();
    return style && style->textShadow();
}

LineDecorationStyle AccessibilityObject::lineDecorationStyle() const
{
    CheckedPtr renderer = this->renderer();
    return renderer ? LineDecorationStyle(*renderer) : LineDecorationStyle();
}

AttributedStringStyle AccessibilityObject::stylesForAttributedString() const
{
    const CheckedPtr style = this->style();
    if (!style)
        return { };

    auto alignment = style->verticalAlign();
    return {
        fontFrom(*style),
        textColorFrom(*style),
        backgroundColorFrom(*style),
        alignment == VerticalAlign::Sub,
        alignment == VerticalAlign::Super,
        !!style->textShadow(),
        lineDecorationStyle()
    };
}

} // namespace WebCore

#endif // PLATFORM(COCOA)
