/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 28, 2022.
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
#include "XMLTreeViewer.h"

#if ENABLE(XSLT)

#include "CSSPrimitiveValue.h"
#include "Document.h"
#include "Element.h"
#include "FrameDestructionObserverInlines.h"
#include "LocalFrame.h"
#include "ScriptController.h"
#include "ScriptSourceCode.h"
#include "SecurityOrigin.h"
#include "SecurityOriginPolicy.h"
#include "StyleScope.h"
#include "Text.h"
#include "XMLViewerCSS.h"
#include "XMLViewerJS.h"

namespace WebCore {

XMLTreeViewer::XMLTreeViewer(Document& document)
    : m_document(document)
{
}

void XMLTreeViewer::transformDocumentToTreeView()
{
    String scriptString = StringImpl::createWithoutCopying(XMLViewer_js);
    Ref document = m_document.get();
    RefPtr frame = document->frame();
    frame->checkedScript()->evaluateIgnoringException(ScriptSourceCode(scriptString, JSC::SourceTaintedOrigin::Untainted));
    frame->checkedScript()->evaluateIgnoringException(ScriptSourceCode(AtomString("prepareWebKitXMLViewer('This XML file does not appear to have any style information associated with it. The document tree is shown below.');"_s), JSC::SourceTaintedOrigin::Untainted));

    String cssString = StringImpl::createWithoutCopying(XMLViewer_css);
    Ref text = document->createTextNode(WTFMove(cssString));
    document->getElementById(String("xml-viewer-style"_s))->appendChild(WTFMove(text));
}

} // namespace WebCore

#endif // ENABLE(XSLT)
