/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 2, 2022.
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
#include "config.h"
#include "SVGDocument.h"

#include "DocumentSVG.h"
#include "SVGSVGElement.h"
#include "SVGViewSpec.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGDocument);

SVGDocument::SVGDocument(LocalFrame* frame, const Settings& settings, const URL& url)
    : XMLDocument(frame, settings, url, { DocumentClass::SVG })
{
}

bool SVGDocument::zoomAndPanEnabled() const
{
    RefPtr element = DocumentSVG::rootElement(*this);
    if (!element)
        return false;
    return (element->useCurrentView() ? element->currentView().zoomAndPan() : element->zoomAndPan()) == SVGZoomAndPanMagnify;
}

void SVGDocument::startPan(const FloatPoint& start)
{
    RefPtr element = DocumentSVG::rootElement(*this);
    if (!element)
        return;
    m_panningOffset = start - element->currentTranslateValue();
}

void SVGDocument::updatePan(const FloatPoint& position) const
{
    RefPtr element = DocumentSVG::rootElement(*this);
    if (!element)
        return;
    element->setCurrentTranslate(position - m_panningOffset);
}

Ref<Document> SVGDocument::cloneDocumentWithoutChildren() const
{
    return create(nullptr, protectedSettings(), url());
}

}
