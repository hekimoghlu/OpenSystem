/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 15, 2023.
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
#include "ModelDocument.h"

#if ENABLE(MODEL_ELEMENT)

#include "Document.h"
#include "DocumentLoader.h"
#include "EventNames.h"
#include "FrameLoader.h"
#include "HTMLBodyElement.h"
#include "HTMLHeadElement.h"
#include "HTMLHtmlElement.h"
#include "HTMLMetaElement.h"
#include "HTMLModelElement.h"
#include "HTMLNames.h"
#include "HTMLSourceElement.h"
#include "HTMLStyleElement.h"
#include "LocalFrame.h"
#include "LocalFrameLoaderClient.h"
#include "RawDataDocumentParser.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ModelDocument);

using namespace HTMLNames;

class ModelDocumentParser final : public RawDataDocumentParser {
public:
    static Ref<ModelDocumentParser> create(ModelDocument& document)
    {
        return adoptRef(*new ModelDocumentParser(document));
    }

private:
    ModelDocumentParser(ModelDocument& document)
        : RawDataDocumentParser { document }
        , m_outgoingReferrer { document.outgoingReferrer() }
    {
    }

    void createDocumentStructure();

    void appendBytes(DocumentWriter&, std::span<const uint8_t>) final;
    void finish() final;

    WeakPtr<HTMLModelElement, WeakPtrImplWithEventTargetData> m_modelElement;
    String m_outgoingReferrer;
};

void ModelDocumentParser::createDocumentStructure()
{
    auto& document = *this->document();

    auto rootElement = HTMLHtmlElement::create(document);
    document.appendChild(rootElement);
    document.setCSSTarget(rootElement.ptr());

    if (document.frame())
        document.frame()->injectUserScripts(UserScriptInjectionTime::DocumentStart);

    auto headElement = HTMLHeadElement::create(document);
    rootElement->appendChild(headElement);

    auto metaElement = HTMLMetaElement::create(document);
    metaElement->setAttributeWithoutSynchronization(nameAttr, "viewport"_s);
    metaElement->setAttributeWithoutSynchronization(contentAttr, "width=device-width,initial-scale=1"_s);
    headElement->appendChild(metaElement);

    auto styleElement = HTMLStyleElement::create(document);
    auto styleContent = "body { background-color: white; text-align: center; }\n"
        "@media (prefers-color-scheme: dark) { body { background-color: rgb(32, 32, 37); } }\n"
        "model { width: 80vw; height: 80vh; }\n"_s;
    styleElement->setTextContent(styleContent);
    headElement->appendChild(styleElement);

    auto body = HTMLBodyElement::create(document);
    rootElement->appendChild(body);

    auto modelElement = HTMLModelElement::create(HTMLNames::modelTag, document);
    m_modelElement = modelElement.get();
    modelElement->setAttributeWithoutSynchronization(interactiveAttr, emptyAtom());

    auto sourceElement = HTMLSourceElement::create(HTMLNames::sourceTag, document);
    sourceElement->setAttributeWithoutSynchronization(srcAttr, AtomString { document.url().string() });
    if (RefPtr loader = document.loader())
        sourceElement->setAttributeWithoutSynchronization(typeAttr, AtomString { loader->responseMIMEType() });

    modelElement->appendChild(sourceElement);

    body->appendChild(modelElement);
    document.setHasVisuallyNonEmptyCustomContent();

    auto frame = document.frame();
    if (!frame)
        return;

    frame->loader().activeDocumentLoader()->setMainResourceDataBufferingPolicy(DataBufferingPolicy::DoNotBufferData);
    frame->loader().setOutgoingReferrer(document.completeURL(m_outgoingReferrer));
}

void ModelDocumentParser::appendBytes(DocumentWriter&, std::span<const uint8_t>)
{
    if (!m_modelElement)
        createDocumentStructure();
}

void ModelDocumentParser::finish()
{
    document()->finishedParsing();
}

ModelDocument::ModelDocument(LocalFrame* frame, const Settings& settings, const URL& url)
    : HTMLDocument(frame, settings, url, { }, { DocumentClass::Model })
{
    if (frame)
        m_outgoingReferrer = frame->loader().outgoingReferrer();
}

Ref<DocumentParser> ModelDocument::createParser()
{
    return ModelDocumentParser::create(*this);
}

}

#endif
