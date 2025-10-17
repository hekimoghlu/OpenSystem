/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 26, 2024.
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
#import "PDFPluginAnnotation.h"

#if ENABLE(PDF_PLUGIN)

#import "PDFAnnotationTypeHelpers.h"
#import "PDFPluginBase.h"
#import "PDFPluginChoiceAnnotation.h"
#import "PDFPluginTextAnnotation.h"
#import <CoreGraphics/CoreGraphics.h>
#import <PDFKit/PDFKit.h>
#import <WebCore/AddEventListenerOptions.h>
#import <WebCore/CSSPrimitiveValue.h>
#import <WebCore/CSSPropertyNames.h>
#import <WebCore/Event.h>
#import <WebCore/EventLoop.h>
#import <WebCore/EventNames.h>
#import <WebCore/HTMLInputElement.h>
#import <WebCore/HTMLNames.h>
#import <WebCore/HTMLOptionElement.h>
#import <WebCore/HTMLSelectElement.h>
#import <WebCore/HTMLTextAreaElement.h>
#import <WebCore/Page.h>

#import "PDFKitSoftLink.h"

namespace WebKit {
using namespace WebCore;
using namespace HTMLNames;
using namespace WebKit::PDFAnnotationTypeHelpers;

RefPtr<PDFPluginAnnotation> PDFPluginAnnotation::create(PDFAnnotation *annotation, PDFPluginBase* plugin)
{
    if (annotationIsWidgetOfType(annotation, WidgetType::Text))
        return PDFPluginTextAnnotation::create(annotation, plugin);
#if PLATFORM(MAC)
    if (annotationIsWidgetOfType(annotation, WidgetType::Choice))
        return PDFPluginChoiceAnnotation::create(annotation, plugin);
#endif

    return nullptr;
}

void PDFPluginAnnotation::attach(Element* parent)
{
    ASSERT(!m_parent);

    m_parent = parent;
    Ref element = createAnnotationElement();
    m_element = element.copyRef();

    if (!element->hasClass())
        element->setAttributeWithoutSynchronization(classAttr, "annotation"_s);
    element->setAttributeWithoutSynchronization(x_apple_pdf_annotationAttr, "true"_s);
    element->addEventListener(eventNames().changeEvent, *m_eventListener, false);
    element->addEventListener(eventNames().blurEvent, *m_eventListener, false);

    updateGeometry();

    RefPtr { m_parent.get() }->appendChild(element);

    // FIXME: The text cursor doesn't blink after this. Why?
    element->focus();
}

void PDFPluginAnnotation::commit()
{
    m_plugin->didMutatePDFDocument();
}

PDFPluginAnnotation::~PDFPluginAnnotation()
{
    m_element->removeEventListener(eventNames().changeEvent, *m_eventListener, false);
    m_element->removeEventListener(eventNames().blurEvent, *m_eventListener, false);

    m_eventListener->setAnnotation(nullptr);

    m_element->document().eventLoop().queueTask(TaskSource::InternalAsyncTask, [ weakElement = WeakPtr<Node, WeakPtrImplWithEventTargetData> { element() } ]() {
        if (RefPtr element = weakElement.get())
            element->remove();
    });
}

void PDFPluginAnnotation::updateGeometry()
{
    auto annotationRect = m_plugin->pluginBoundsForAnnotation(m_annotation);

    Ref styledElement = downcast<StyledElement>(*element());
    styledElement->setInlineStyleProperty(CSSPropertyWidth, annotationRect.size.width, CSSUnitType::CSS_PX);
    styledElement->setInlineStyleProperty(CSSPropertyHeight, annotationRect.size.height, CSSUnitType::CSS_PX);
    styledElement->setInlineStyleProperty(CSSPropertyLeft, annotationRect.origin.x, CSSUnitType::CSS_PX);
    styledElement->setInlineStyleProperty(CSSPropertyTop, annotationRect.origin.y, CSSUnitType::CSS_PX);
}

bool PDFPluginAnnotation::handleEvent(Event& event)
{
    if (event.type() == eventNames().blurEvent || event.type() == eventNames().changeEvent) {
        m_plugin->setActiveAnnotation({ nullptr });
        return true;
    }

    return false;
}

void PDFPluginAnnotation::PDFPluginAnnotationEventListener::handleEvent(ScriptExecutionContext&, Event& event)
{
    if (m_annotation)
        m_annotation->handleEvent(event);
}

} // namespace WebKit

#endif // ENABLE(PDF_PLUGIN)
