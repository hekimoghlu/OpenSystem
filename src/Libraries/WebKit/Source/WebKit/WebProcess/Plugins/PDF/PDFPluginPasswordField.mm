/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 30, 2024.
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
#import "PDFPluginPasswordField.h"

#if ENABLE(PDF_PLUGIN)

#import <WebCore/AddEventListenerOptions.h>
#import <WebCore/EnterKeyHint.h>
#import <WebCore/Event.h>
#import <WebCore/EventNames.h>
#import <WebCore/HTMLElement.h>
#import <WebCore/KeyboardEvent.h>

namespace WebKit {
using namespace WebCore;
using namespace HTMLNames;

Ref<PDFPluginPasswordField> PDFPluginPasswordField::create(PDFPluginBase* plugin)
{
    return adoptRef(*new PDFPluginPasswordField(plugin));
}

PDFPluginPasswordField::~PDFPluginPasswordField()
{
    element()->removeEventListener(eventNames().keyupEvent, *eventListener(), false);
}

Ref<Element> PDFPluginPasswordField::createAnnotationElement()
{
    auto element = PDFPluginTextAnnotation::createAnnotationElement();
    element->setAttribute(typeAttr, "password"_s);
    element->setAttribute(enterkeyhintAttr, AtomString { attributeValueForEnterKeyHint(EnterKeyHint::Go) });
    element->addEventListener(eventNames().keyupEvent, *eventListener(), false);
    return element;
}

void PDFPluginPasswordField::updateGeometry()
{
    // Intentionally do not call the superclass.
}

bool PDFPluginPasswordField::handleEvent(WebCore::Event& event)
{
    if (auto* keyboardEvent = dynamicDowncast<KeyboardEvent>(event); keyboardEvent && keyboardEvent->type() == eventNames().keyupEvent) {
        if (keyboardEvent->keyIdentifier() == "Enter"_s) {
            plugin()->attemptToUnlockPDF(value());
            event.preventDefault();
            return true;
        }
    }

    return false;
}

void PDFPluginPasswordField::resetField()
{
    setValue(""_s);
}
    
} // namespace WebKit

#endif // ENABLE(PDF_PLUGIN)
