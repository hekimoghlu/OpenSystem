/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 15, 2025.
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
#import "PDFPluginPasswordForm.h"

#if ENABLE(PDF_PLUGIN)

#import <WebCore/ElementInlines.h>
#import <WebCore/HTMLElement.h>
#import <WebCore/SVGNames.h>
#import <WebCore/XMLNSNames.h>

namespace WebKit {
using namespace WebCore;
using namespace HTMLNames;
using namespace SVGNames;
using namespace XMLNSNames;

Ref<PDFPluginPasswordForm> PDFPluginPasswordForm::create(PDFPluginBase* plugin)
{
    return adoptRef(*new PDFPluginPasswordForm(plugin));
}

PDFPluginPasswordForm::~PDFPluginPasswordForm() = default;

Ref<Element> PDFPluginPasswordForm::createAnnotationElement()
{
    Ref document = parent()->document();
    Ref element = document->createElement(divTag, false);
    element->setAttributeWithoutSynchronization(classAttr, "password-form"_s);

    auto iconElement = document->createElement(svgTag, false);
    iconElement->setAttributeWithoutSynchronization(classAttr, "lock-icon"_s);
    iconElement->setAttributeWithoutSynchronization(xmlnsAttr, "http://www.w3.org/2000/svg"_s);
    iconElement->setAttributeWithoutSynchronization(viewBoxAttr, "0 -10 106.51 106.51"_s);
    element->appendChild(iconElement);

    auto iconPathElement = document->createElement(pathTag, false);
    iconPathElement->setAttributeWithoutSynchronization(dAttr, "M30.898 91.338H75.599C82.77 91.338 86.57 87.454 86.57 79.725V46.13C86.57 38.442 82.77 34.558 75.6 34.558H30.898C23.738 34.558 19.938 38.442 19.938 46.13V79.725C19.938 87.454 23.738 91.338 30.898 91.338ZM31.166 84.002C29.084 84.002 27.813 82.678 27.813 80.263V45.592C27.813 43.177 29.083 41.894 31.166 41.894H75.34C77.465 41.894 78.695 43.177 78.695 45.592V80.263C78.695 82.678 77.465 84.002 75.34 84.002ZM28.516 38.309H36.237V21.845C36.236 9.505 44.13 2.984 53.228 2.984 62.356 2.984 70.312 9.504 70.312 21.844V38.31H78.032V22.899C78.032 4.5 66.025-4.351 53.228-4.351 40.47-4.352 28.517 4.5 28.517 22.9Z"_s);
    iconPathElement->setAttributeWithoutSynchronization(fillAttr, "#4B4B4B"_s);
    iconElement->appendChild(iconPathElement);

    auto titleElement = document->createElement(pTag, false);
    titleElement->setTextContent(pdfPasswordFormTitle());
    titleElement->setAttributeWithoutSynchronization(classAttr, "title"_s);
    element->appendChild(titleElement);

    m_subtitleElement = document->createElement(pTag, false);
    m_subtitleElement->setTextContent(pdfPasswordFormSubtitle());
    m_subtitleElement->setAttributeWithoutSynchronization(classAttr, "subtitle"_s);
    element->appendChild(*m_subtitleElement);

    return element;
}

void PDFPluginPasswordForm::unlockFailed()
{
    m_subtitleElement->setTextContent(pdfPasswordFormInvalidPasswordSubtitle());
}

void PDFPluginPasswordForm::updateGeometry()
{
    // Intentionally do not call the superclass.
}

} // namespace WebKit

#endif // ENABLE(PDF_PLUGIN)
