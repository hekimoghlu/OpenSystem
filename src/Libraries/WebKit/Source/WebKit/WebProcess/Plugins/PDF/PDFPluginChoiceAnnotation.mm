/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 15, 2023.
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
#import "PDFPluginChoiceAnnotation.h"

#if ENABLE(PDF_PLUGIN) && PLATFORM(MAC)

#import "PDFAnnotationTypeHelpers.h"
#import "PDFKitSPI.h"
#import <WebCore/CSSPrimitiveValue.h>
#import <WebCore/CSSPropertyNames.h>
#import <WebCore/ColorMac.h>
#import <WebCore/ColorSerialization.h>
#import <WebCore/HTMLElement.h>
#import <WebCore/HTMLNames.h>
#import <WebCore/HTMLOptionElement.h>
#import <WebCore/HTMLSelectElement.h>
#import <WebCore/Page.h>

namespace WebKit {
using namespace WebCore;
using namespace HTMLNames;

Ref<PDFPluginChoiceAnnotation> PDFPluginChoiceAnnotation::create(PDFAnnotation *annotation, PDFPluginBase* plugin)
{
    ASSERT(PDFAnnotationTypeHelpers::annotationIsWidgetOfType(annotation, WidgetType::Choice));
    return adoptRef(*new PDFPluginChoiceAnnotation(annotation, plugin));
}

void PDFPluginChoiceAnnotation::updateGeometry()
{
    PDFPluginAnnotation::updateGeometry();

    Ref styledElement = downcast<StyledElement>(*element());
    styledElement->setInlineStyleProperty(CSSPropertyFontSize, annotation().font.pointSize * plugin()->contentScaleFactor(), CSSUnitType::CSS_PX);
}

void PDFPluginChoiceAnnotation::commit()
{
    annotation().widgetStringValue = downcast<HTMLSelectElement>(element())->value();

    PDFPluginAnnotation::commit();
}

Ref<Element> PDFPluginChoiceAnnotation::createAnnotationElement()
{
    Ref document = parent()->document();
    RetainPtr choiceAnnotation = annotation();

    Ref element = downcast<StyledElement>(document->createElement(selectTag, false));

    // FIXME: Match font weight and style as well?
    element->setInlineStyleProperty(CSSPropertyColor, serializationForHTML(colorFromCocoaColor([choiceAnnotation fontColor])));
    element->setInlineStyleProperty(CSSPropertyFontFamily, [[choiceAnnotation font] familyName]);

    NSArray *choices = [choiceAnnotation choices];
    NSString *selectedChoice = [choiceAnnotation widgetStringValue];

    for (NSString *choice in choices) {
        auto choiceOption = document->createElement(optionTag, false);
        choiceOption->setAttributeWithoutSynchronization(valueAttr, choice);
        choiceOption->setTextContent(choice);

        if (choice == selectedChoice)
            choiceOption->setAttributeWithoutSynchronization(selectedAttr, "selected"_s);

        element->appendChild(choiceOption);
    }

    return element;
}

} // namespace WebKit

#endif // ENABLE(PDF_PLUGIN) && PLATFORM(MAC)
