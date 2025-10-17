/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 3, 2024.
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
#import "FontAttributes.h"

#import "CSSValueKeywords.h"
#import "ColorCocoa.h"
#import "FontCocoa.h"
#import <pal/spi/cocoa/NSAttributedStringSPI.h>
#import <wtf/cocoa/VectorCocoa.h>

#if PLATFORM(IOS_FAMILY)
#import <pal/ios/UIKitSoftLink.h>
#endif

namespace WebCore {

static NSString *cocoaTextListMarkerName(ListStyleType styleType, bool ordered)
{
    if (styleType.type == ListStyleType::Type::CounterStyle && styleType.identifier == nameLiteral(CSSValueDisc))
        return NSTextListMarkerDisc;
    if (styleType.type == ListStyleType::Type::CounterStyle && styleType.identifier == nameLiteral(CSSValueCircle))
        return NSTextListMarkerCircle;
    if (styleType.type == ListStyleType::Type::CounterStyle && styleType.identifier == nameLiteral(CSSValueSquare))
        return NSTextListMarkerSquare;
    if (styleType.type == ListStyleType::Type::CounterStyle && styleType.identifier == nameLiteral(CSSValueDecimal))
        return NSTextListMarkerDecimal;
    if (styleType.type == ListStyleType::Type::CounterStyle && styleType.identifier == nameLiteral(CSSValueOctal))
        return NSTextListMarkerOctal;
    if (styleType.type == ListStyleType::Type::CounterStyle && styleType.identifier == nameLiteral(CSSValueLowerRoman))
        return NSTextListMarkerLowercaseRoman;
    if (styleType.type == ListStyleType::Type::CounterStyle && styleType.identifier == nameLiteral(CSSValueUpperRoman))
        return NSTextListMarkerUppercaseRoman;
    if (styleType.type == ListStyleType::Type::CounterStyle && styleType.identifier == nameLiteral(CSSValueLowerAlpha))
        return NSTextListMarkerLowercaseAlpha;
    if (styleType.type == ListStyleType::Type::CounterStyle && styleType.identifier == nameLiteral(CSSValueUpperAlpha))
        return NSTextListMarkerUppercaseAlpha;
    if (styleType.type == ListStyleType::Type::CounterStyle && styleType.identifier == nameLiteral(CSSValueLowerLatin))
        return NSTextListMarkerLowercaseLatin;
    if (styleType.type == ListStyleType::Type::CounterStyle && styleType.identifier == nameLiteral(CSSValueUpperLatin))
        return NSTextListMarkerUppercaseLatin;
    if (styleType.type == ListStyleType::Type::CounterStyle && styleType.identifier == nameLiteral(CSSValueLowerHexadecimal))
        return NSTextListMarkerLowercaseHexadecimal;
    if (styleType.type == ListStyleType::Type::CounterStyle && styleType.identifier == nameLiteral(CSSValueUpperHexadecimal))
        return NSTextListMarkerUppercaseHexadecimal;
    // The remaining web-exposed list style types have no Cocoa equivalents.
    // Fall back to default styles for ordered and unordered lists.
    return ordered ? NSTextListMarkerDecimal : NSTextListMarkerDisc;
}

RetainPtr<NSTextList> TextList::createTextList() const
{
#if PLATFORM(MAC)
    Class textListClass = NSTextList.class;
#else
    Class textListClass = PAL::getNSTextListClass();
#endif
    auto result = adoptNS([[textListClass alloc] initWithMarkerFormat:cocoaTextListMarkerName(styleType, ordered) options:0]);
    [result setStartingItemNumber:startingItemNumber];
    return result;
}

RetainPtr<NSDictionary> FontAttributes::createDictionary() const
{
    NSMutableDictionary *attributes = [NSMutableDictionary dictionary];
    if (auto cocoaFont = font ? (__bridge CocoaFont *)font->getCTFont() : nil)
        attributes[NSFontAttributeName] = cocoaFont;

    if (foregroundColor.isValid())
        attributes[NSForegroundColorAttributeName] = cocoaColor(foregroundColor).get();

    if (backgroundColor.isValid())
        attributes[NSBackgroundColorAttributeName] = cocoaColor(backgroundColor).get();

    if (fontShadow.color.isValid() && (!fontShadow.offset.isZero() || fontShadow.blurRadius))
        attributes[NSShadowAttributeName] = fontShadow.createShadow().get();

    if (subscriptOrSuperscript == SubscriptOrSuperscript::Subscript)
        attributes[NSSuperscriptAttributeName] = @(-1);
    else if (subscriptOrSuperscript == SubscriptOrSuperscript::Superscript)
        attributes[NSSuperscriptAttributeName] = @1;

#if PLATFORM(MAC)
    Class paragraphStyleClass = NSParagraphStyle.class;
#else
    Class paragraphStyleClass = PAL::getNSParagraphStyleClass();
#endif
    auto style = adoptNS([[paragraphStyleClass defaultParagraphStyle] mutableCopy]);

    switch (horizontalAlignment) {
    case HorizontalAlignment::Left:
        [style setAlignment:NSTextAlignmentLeft];
        break;
    case HorizontalAlignment::Center:
        [style setAlignment:NSTextAlignmentCenter];
        break;
    case HorizontalAlignment::Right:
        [style setAlignment:NSTextAlignmentRight];
        break;
    case HorizontalAlignment::Justify:
        [style setAlignment:NSTextAlignmentJustified];
        break;
    case HorizontalAlignment::Natural:
        [style setAlignment:NSTextAlignmentNatural];
        break;
    }

    if (!textLists.isEmpty()) {
        [style setTextLists:createNSArray(textLists, [] (auto& textList) {
            return textList.createTextList();
        }).get()];
    }

    attributes[NSParagraphStyleAttributeName] = style.get();

    if (hasUnderline)
        attributes[NSUnderlineStyleAttributeName] = @(NSUnderlineStyleSingle);

    if (hasStrikeThrough)
        attributes[NSStrikethroughStyleAttributeName] = @(NSUnderlineStyleSingle);

    return attributes;
}

} // namespace WebCore
