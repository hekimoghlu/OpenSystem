/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 22, 2022.
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
#include "PluginDocument.h"

#include "DocumentLoader.h"
#include "FrameLoader.h"
#include "HTMLBodyElement.h"
#include "HTMLEmbedElement.h"
#include "HTMLHeadElement.h"
#include "HTMLHtmlElement.h"
#include "HTMLNames.h"
#include "HTMLStyleElement.h"
#include "LocalFrame.h"
#include "LocalFrameLoaderClient.h"
#include "LocalFrameView.h"
#include "Logging.h"
#include "PluginViewBase.h"
#include "RawDataDocumentParser.h"
#include "RenderEmbeddedObject.h"
#include "StyleSheetContents.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(PluginDocument);

using namespace HTMLNames;

// FIXME: Share more code with MediaDocumentParser.
class PluginDocumentParser final : public RawDataDocumentParser {
public:
    static Ref<PluginDocumentParser> create(PluginDocument& document)
    {
        return adoptRef(*new PluginDocumentParser(document));
    }

private:
    PluginDocumentParser(Document& document)
        : RawDataDocumentParser(document)
    {
    }

    void appendBytes(DocumentWriter&, std::span<const uint8_t>) final;
    void createDocumentStructure();
    static Ref<HTMLStyleElement> createStyleElement(Document&);

    WeakPtr<HTMLEmbedElement, WeakPtrImplWithEventTargetData> m_embedElement;
};

Ref<HTMLStyleElement> PluginDocumentParser::createStyleElement(Document& document)
{
    auto styleElement = HTMLStyleElement::create(document);

    constexpr auto styleSheetContents = R"CONTENTS(
        html, body, embed { width: 100%; height: 100%; }
        body { margin: 0; overflow: hidden; }
        html.plugin-fits-content body { overflow: revert; }
    )CONTENTS"_s;
#if PLATFORM(IOS_FAMILY)
    constexpr auto bodyBackgroundColorStyle = "body { background-color: rgb(217, 224, 233) }"_s;
#else
    constexpr auto bodyBackgroundColorStyle = "body { background-color: rgb(38, 38, 38) }"_s;
#endif
    styleElement->setTextContent(makeString(styleSheetContents, bodyBackgroundColorStyle));
    return styleElement;
}

void PluginDocumentParser::createDocumentStructure()
{
    auto& document = downcast<PluginDocument>(*this->document());

    LOG_WITH_STREAM(Plugins, stream << "PluginDocumentParser::createDocumentStructure() for document " << document);

    auto rootElement = HTMLHtmlElement::create(document);
    document.appendChild(rootElement);

    auto headElement = HTMLHeadElement::create(document);
    auto styleElement = createStyleElement(document);
    headElement->appendChild(styleElement);
    rootElement->appendChild(headElement);

    if (document.frame())
        document.frame()->injectUserScripts(UserScriptInjectionTime::DocumentStart);

#if PLATFORM(IOS_FAMILY)
    // Should not be able to zoom into standalone plug-in documents.
    document.processViewport("user-scalable=no"_s, ViewportArguments::Type::PluginDocument);
#endif

    auto body = HTMLBodyElement::create(document);
    rootElement->appendChild(body);
        
    auto embedElement = HTMLEmbedElement::create(document);
    m_embedElement = embedElement.get();
    embedElement->setAttributeWithoutSynchronization(nameAttr, "plugin"_s);
    embedElement->setAttributeWithoutSynchronization(srcAttr, AtomString { document.url().string() });
    
    ASSERT(document.loader());
    if (RefPtr loader = document.loader())
        m_embedElement->setAttributeWithoutSynchronization(typeAttr, AtomString { loader->writer().mimeType() });

    document.setPluginElement(*m_embedElement);

    body->appendChild(embedElement);
    document.setHasVisuallyNonEmptyCustomContent();
}

void PluginDocumentParser::appendBytes(DocumentWriter&, std::span<const uint8_t>)
{
    if (m_embedElement)
        return;

    createDocumentStructure();

    RefPtr frame = document()->frame();
    if (!frame)
        return;

    document()->updateLayout();

    // Below we assume that renderer->widget() to have been created by
    // document()->updateLayout(). However, in some cases, updateLayout() will 
    // recurse too many times and delay its post-layout tasks (such as creating
    // the widget). Here we kick off the pending post-layout tasks so that we
    // can synchronously redirect data to the plugin.
    frame->view()->flushAnyPendingPostLayoutTasks();

    if (auto renderer = m_embedElement->renderWidget()) {
        if (RefPtr widget = renderer->widget()) {
            frame->loader().client().redirectDataToPlugin(*widget);

            // In a plugin document, the main resource is the plugin. If we have a null widget, that means
            // the loading of the plugin was cancelled, which gives us a null mainResourceLoader(), so we
            // need to have this call in a null check of the widget or of mainResourceLoader().
            if (auto loader = frame->loader().activeDocumentLoader())
                loader->setMainResourceDataBufferingPolicy(DataBufferingPolicy::DoNotBufferData);
        }
    }
}

PluginDocument::PluginDocument(LocalFrame& frame, const URL& url)
    : HTMLDocument(&frame, frame.settings(), url, { }, { DocumentClass::Plugin })
{
    setCompatibilityMode(DocumentCompatibilityMode::NoQuirksMode);
    lockCompatibilityMode();
}

PluginDocument::~PluginDocument() = default;

Ref<DocumentParser> PluginDocument::createParser()
{
    return PluginDocumentParser::create(*this);
}

PluginViewBase* PluginDocument::pluginWidget()
{
    if (!m_pluginElement)
        return nullptr;
    auto* renderer = dynamicDowncast<RenderEmbeddedObject>(m_pluginElement->renderer());
    if (!renderer)
        return nullptr;
    return dynamicDowncast<PluginViewBase>(renderer->widget());
}

void PluginDocument::setPluginElement(HTMLPlugInElement& element)
{
    m_pluginElement = &element;
}

void PluginDocument::detachFromPluginElement()
{
    // Release the plugin Element so that we don't have a circular reference.
    m_pluginElement = nullptr;
}

void PluginDocument::releaseMemory()
{
    if (RefPtr pluginView = pluginWidget())
        pluginView->releaseMemory();
}

}
