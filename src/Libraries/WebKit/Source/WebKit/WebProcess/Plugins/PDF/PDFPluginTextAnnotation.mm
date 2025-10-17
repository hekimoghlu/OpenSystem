/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 10, 2024.
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
#import "PDFPluginTextAnnotation.h"

#if ENABLE(PDF_PLUGIN)

#import "PDFAnnotationTypeHelpers.h"
#import "PDFKitSPI.h"
#import <WebCore/AddEventListenerOptions.h>
#import <WebCore/CSSPrimitiveValue.h>
#import <WebCore/CSSPropertyNames.h>
#import <WebCore/ColorCocoa.h>
#import <WebCore/ColorSerialization.h>
#import <WebCore/Event.h>
#import <WebCore/EventNames.h>
#import <WebCore/HTMLElement.h>
#import <WebCore/HTMLInputElement.h>
#import <WebCore/HTMLNames.h>
#import <WebCore/HTMLTextAreaElement.h>
#import <WebCore/KeyboardEvent.h>
#import <WebCore/Page.h>

namespace WebKit {
using namespace WebCore;
using namespace HTMLNames;

static const String cssAlignmentValueForNSTextAlignment(NSTextAlignment alignment)
{
    switch (alignment) {
    case NSTextAlignmentLeft:
        return "left"_s;
    case NSTextAlignmentRight:
        return "right"_s;
    case NSTextAlignmentCenter:
        return "center"_s;
    case NSTextAlignmentJustified:
        return "justify"_s;
    case NSTextAlignmentNatural:
        return "-webkit-start"_s;
    }
    ASSERT_NOT_REACHED();
    return String();
}

Ref<PDFPluginTextAnnotation> PDFPluginTextAnnotation::create(PDFAnnotation *annotation, PDFPluginBase* plugin)
{
    ASSERT(PDFAnnotationTypeHelpers::annotationIsWidgetOfType(annotation, WidgetType::Text));
    return adoptRef(*new PDFPluginTextAnnotation(annotation, plugin));
}

PDFPluginTextAnnotation::~PDFPluginTextAnnotation()
{
    element()->removeEventListener(eventNames().keydownEvent, *eventListener(), false);
}

Ref<Element> PDFPluginTextAnnotation::createAnnotationElement()
{
    Document& document = parent()->document();
    RetainPtr textAnnotation = annotation();
    bool isMultiline = [textAnnotation isMultiline];

    Ref element = downcast<HTMLTextFormControlElement>(document.createElement(isMultiline ? textareaTag : inputTag, false));
    element->addEventListener(eventNames().keydownEvent, *eventListener(), false);

    if (!textAnnotation)
        return element;

    // FIXME: Match font weight and style as well?
    element->setInlineStyleProperty(CSSPropertyColor, serializationForHTML(colorFromCocoaColor([textAnnotation fontColor])));
    element->setInlineStyleProperty(CSSPropertyFontFamily, [[textAnnotation font] familyName]);
    element->setInlineStyleProperty(CSSPropertyTextAlign, cssAlignmentValueForNSTextAlignment([textAnnotation alignment]));

    element->setValue([textAnnotation widgetStringValue]);

    return element;
}

void PDFPluginTextAnnotation::updateGeometry()
{
    PDFPluginAnnotation::updateGeometry();

    Ref styledElement = downcast<StyledElement>(*element());
    styledElement->setInlineStyleProperty(CSSPropertyFontSize, annotation().font.pointSize * plugin()->contentScaleFactor(), CSSUnitType::CSS_PX);
}

void PDFPluginTextAnnotation::commit()
{
    annotation().widgetStringValue = value();
    PDFPluginAnnotation::commit();
}

String PDFPluginTextAnnotation::value() const
{
    return downcast<HTMLTextFormControlElement>(element())->value();
}

void PDFPluginTextAnnotation::setValue(const String& value)
{
    downcast<HTMLTextFormControlElement>(element())->setValue(value);
}

bool PDFPluginTextAnnotation::handleEvent(Event& event)
{
    if (PDFPluginAnnotation::handleEvent(event))
        return true;

    if (auto* keyboardEvent = dynamicDowncast<KeyboardEvent>(event); keyboardEvent && keyboardEvent->type() == eventNames().keydownEvent) {
        if (keyboardEvent->keyIdentifier() == "U+0009"_s) {
            if (keyboardEvent->ctrlKey() || keyboardEvent->metaKey())
                return false;

            if (keyboardEvent->shiftKey())
                plugin()->focusPreviousAnnotation();
            else
                plugin()->focusNextAnnotation();
            
            event.preventDefault();
            return true;
        }
    }

    return false;
}

} // namespace WebKit

#endif // ENABLE(PDF_PLUGIN)
