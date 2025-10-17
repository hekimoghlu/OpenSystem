/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 21, 2023.
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
#include "MediaDocument.h"

#if ENABLE(VIDEO)

#include "Chrome.h"
#include "ChromeClient.h"
#include "DocumentLoader.h"
#include "EventNames.h"
#include "FrameLoader.h"
#include "HTMLBodyElement.h"
#include "HTMLEmbedElement.h"
#include "HTMLHeadElement.h"
#include "HTMLHtmlElement.h"
#include "HTMLMetaElement.h"
#include "HTMLNames.h"
#include "HTMLSourceElement.h"
#include "HTMLVideoElement.h"
#include "KeyboardEvent.h"
#include "LocalFrame.h"
#include "LocalFrameLoaderClient.h"
#include "MouseEvent.h"
#include "NodeList.h"
#include "Page.h"
#include "RawDataDocumentParser.h"
#include "ScriptController.h"
#include "ShadowRoot.h"
#include "TypedElementDescendantIteratorInlines.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(MediaDocument);

using namespace HTMLNames;

// FIXME: Share more code with PluginDocumentParser.
class MediaDocumentParser final : public RawDataDocumentParser {
public:
    static Ref<MediaDocumentParser> create(MediaDocument& document)
    {
        return adoptRef(*new MediaDocumentParser(document));
    }
    
private:
    MediaDocumentParser(MediaDocument& document)
        : RawDataDocumentParser { document }
        , m_outgoingReferrer { document.outgoingReferrer() }
    {
    }

    void appendBytes(DocumentWriter&, std::span<const uint8_t>) final;
    void createDocumentStructure();

    WeakPtr<HTMLMediaElement> m_mediaElement;
    String m_outgoingReferrer;
};
    
void MediaDocumentParser::createDocumentStructure()
{
    Ref document = *this->document();

    Ref rootElement = HTMLHtmlElement::create(document);
    document->appendChild(rootElement);
    document->setCSSTarget(rootElement.ptr());

    if (RefPtr frame = document->frame())
        frame->injectUserScripts(UserScriptInjectionTime::DocumentStart);

#if PLATFORM(IOS_FAMILY)
    Ref headElement = HTMLHeadElement::create(document);
    rootElement->appendChild(headElement);

    Ref metaElement = HTMLMetaElement::create(document);
    metaElement->setAttributeWithoutSynchronization(nameAttr, "viewport"_s);
    metaElement->setAttributeWithoutSynchronization(contentAttr, "width=device-width,initial-scale=1"_s);
    headElement->appendChild(metaElement);
#endif

    Ref body = HTMLBodyElement::create(document);
    rootElement->appendChild(body);

    Ref videoElement = HTMLVideoElement::create(document);
    m_mediaElement = videoElement.get();
    videoElement->setAttributeWithoutSynchronization(controlsAttr, emptyAtom());
    videoElement->setAttributeWithoutSynchronization(autoplayAttr, emptyAtom());
    videoElement->setAttributeWithoutSynchronization(srcAttr, AtomString { document->url().string() });
    if (RefPtr loader = document->loader())
        videoElement->setAttributeWithoutSynchronization(typeAttr, AtomString { loader->responseMIMEType() });

    body->appendChild(videoElement);
    document->setHasVisuallyNonEmptyCustomContent();

    RefPtr frame = document->frame();
    if (!frame)
        return;

    frame->loader().protectedActiveDocumentLoader()->setMainResourceDataBufferingPolicy(DataBufferingPolicy::DoNotBufferData);
    frame->protectedLoader()->setOutgoingReferrer(document->completeURL(m_outgoingReferrer));
}

void MediaDocumentParser::appendBytes(DocumentWriter&, std::span<const uint8_t>)
{
    if (m_mediaElement)
        return;

    createDocumentStructure();
    finish();
}
    
MediaDocument::MediaDocument(LocalFrame* frame, const Settings& settings, const URL& url)
    : HTMLDocument(frame, settings, url, { }, { DocumentClass::Media })
{
    setCompatibilityMode(DocumentCompatibilityMode::NoQuirksMode);
    lockCompatibilityMode();
    if (frame)
        m_outgoingReferrer = frame->loader().outgoingReferrer();
}

MediaDocument::~MediaDocument() = default;

Ref<DocumentParser> MediaDocument::createParser()
{
    return MediaDocumentParser::create(*this);
}

static inline HTMLVideoElement* descendantVideoElement(ContainerNode& node)
{
    if (auto* video = dynamicDowncast<HTMLVideoElement>(node))
        return video;

    return descendantsOfType<HTMLVideoElement>(node).first();
}

void MediaDocument::replaceMediaElementTimerFired()
{
    RefPtr htmlBody = bodyOrFrameset();
    if (!htmlBody)
        return;

    // Set body margin width and height to 0 as that is what a PluginDocument uses.
    htmlBody->setAttributeWithoutSynchronization(marginwidthAttr, "0"_s);
    htmlBody->setAttributeWithoutSynchronization(marginheightAttr, "0"_s);

    if (RefPtr videoElement = descendantVideoElement(*htmlBody)) {
        auto embedElement = HTMLEmbedElement::create(*this);

        embedElement->setAttributeWithoutSynchronization(widthAttr, "100%"_s);
        embedElement->setAttributeWithoutSynchronization(heightAttr, "100%"_s);
        embedElement->setAttributeWithoutSynchronization(nameAttr, "plugin"_s);
        embedElement->setAttributeWithoutSynchronization(srcAttr, AtomString { url().string() });

        ASSERT(loader());
        if (RefPtr loader = this->loader())
            embedElement->setAttributeWithoutSynchronization(typeAttr, AtomString { loader->writer().mimeType() });

        videoElement->parentNode()->replaceChild(embedElement, *videoElement);
    }
}

void MediaDocument::mediaElementNaturalSizeChanged(const IntSize& newSize)
{
    if (ownerElement())
        return;

    if (newSize.isZero())
        return;

    if (page())
        page()->chrome().client().imageOrMediaDocumentSizeChanged(newSize);
}

}

#endif
