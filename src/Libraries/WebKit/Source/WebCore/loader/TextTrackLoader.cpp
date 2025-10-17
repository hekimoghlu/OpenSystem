/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 12, 2021.
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

#if ENABLE(VIDEO)

#include "TextTrackLoader.h"

#include "CachedResourceLoader.h"
#include "CachedResourceRequest.h"
#include "CachedTextTrack.h"
#include "CrossOriginAccessControl.h"
#include "DocumentInlines.h"
#include "HTMLMediaElement.h"
#include "HTMLTrackElement.h"
#include "InspectorInstrumentation.h"
#include "Logging.h"
#include "SharedBuffer.h"
#include "VTTCue.h"
#include "WebVTTParser.h"
#include <wtf/NeverDestroyed.h>

namespace WebCore {
    
TextTrackLoader::TextTrackLoader(TextTrackLoaderClient& client, Document& document)
    : m_client(client)
    , m_document(document)
    , m_cueLoadTimer(*this, &TextTrackLoader::cueLoadTimerFired)
{
}

TextTrackLoader::~TextTrackLoader()
{
    if (CachedResourceHandle resource = m_resource)
        resource->removeClient(*this);
}

void TextTrackLoader::cueLoadTimerFired()
{
    if (m_newCuesAvailable) {
        m_newCuesAvailable = false;
        m_client->newCuesAvailable(*this);
    }

    if (m_state >= Finished)
        m_client->cueLoadingCompleted(*this, m_state == Failed);
}

void TextTrackLoader::cancelLoad()
{
    if (CachedResourceHandle resource = std::exchange(m_resource, nullptr))
        resource->removeClient(*this);
}

void TextTrackLoader::processNewCueData(CachedResource& resource)
{
    ASSERT_UNUSED(resource, m_resource == &resource);

    if (m_state == Failed || !m_resource->resourceBuffer())
        return;

    Ref buffer = *m_resource->resourceBuffer();
    if (m_parseOffset == buffer->size())
        return;

    if (!m_cueParser)
        m_cueParser = makeUnique<WebVTTParser>(static_cast<WebVTTParserClient&>(*this), protectedDocument());

    while (m_parseOffset < buffer->size()) {
        auto data = buffer->getSomeData(m_parseOffset);
        m_cueParser->parseBytes(data.span());
        m_parseOffset += data.size();
    }
}

// FIXME: This is a very unusual pattern, no other CachedResourceClient does this. Refactor to use notifyFinished() instead.
void TextTrackLoader::deprecatedDidReceiveCachedResource(CachedResource& resource)
{
    ASSERT_UNUSED(resource, m_resource == &resource);

    if (!m_resource->resourceBuffer())
        return;

    processNewCueData(*protectedResource());
}

void TextTrackLoader::corsPolicyPreventedLoad()
{
    static NeverDestroyed<String> consoleMessage(MAKE_STATIC_STRING_IMPL("Cross-origin text track load denied by Cross-Origin Resource Sharing policy."));
    protectedDocument()->addConsoleMessage(MessageSource::Security, MessageLevel::Error, consoleMessage);
    m_state = Failed;
}

void TextTrackLoader::notifyFinished(CachedResource& resource, const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess)
{
    ASSERT_UNUSED(resource, m_resource == &resource);

    if (m_resource->resourceError().isAccessControl())
        corsPolicyPreventedLoad();

    if (!m_resource->resourceBuffer())
        m_state = Failed;

    if (m_state != Failed) {
        CachedResourceHandle resource = m_resource;
        processNewCueData(*resource);
        if (m_cueParser)
            m_cueParser->fileFinished();
        if (m_state != Failed)
            m_state = resource->errorOccurred() ? Failed : Finished;
    }

    if (m_state == Finished && m_cueParser)
        m_cueParser->flush();

    if (!m_cueLoadTimer.isActive())
        m_cueLoadTimer.startOneShot(0_s);

    cancelLoad();
}

bool TextTrackLoader::load(const URL& url, HTMLTrackElement& element)
{
    cancelLoad();

    ResourceLoaderOptions options = CachedResourceLoader::defaultCachedResourceOptions();
    options.contentSecurityPolicyImposition = element.isInUserAgentShadowTree() ? ContentSecurityPolicyImposition::SkipPolicyCheck : ContentSecurityPolicyImposition::DoPolicyCheck;

    Ref document = m_document.get();
    // FIXME: Do we really need to call completeURL here?
    ResourceRequest resourceRequest(document->completeURL(url.string()));

    if (RefPtr mediaElement = element.mediaElement())
        resourceRequest.setInspectorInitiatorNodeIdentifier(InspectorInstrumentation::identifierForNode(*mediaElement));

    auto cueRequest = createPotentialAccessControlRequest(WTFMove(resourceRequest), WTFMove(options), document, element.mediaElementCrossOriginAttribute());
    m_resource = document->protectedCachedResourceLoader()->requestTextTrack(WTFMove(cueRequest)).value_or(nullptr);
    if (CachedResourceHandle resource = m_resource) {
        resource->addClient(*this);
        return true;
    }
    return false;
}

void TextTrackLoader::newCuesParsed()
{
    if (m_cueLoadTimer.isActive())
        return;

    m_newCuesAvailable = true;
    m_cueLoadTimer.startOneShot(0_s);
}

void TextTrackLoader::newRegionsParsed()
{
    m_client->newRegionsAvailable(*this);
}

void TextTrackLoader::newStyleSheetsParsed()
{
    m_client->newStyleSheetsAvailable(*this);
}

void TextTrackLoader::fileFailedToParse()
{
    LOG(Media, "TextTrackLoader::fileFailedToParse");

    m_state = Failed;

    if (!m_cueLoadTimer.isActive())
        m_cueLoadTimer.startOneShot(0_s);

    cancelLoad();
}

Vector<Ref<VTTCue>> TextTrackLoader::getNewCues()
{
    ASSERT(m_cueParser);
    if (!m_cueParser)
        return { };

    return m_cueParser->takeCues().map([this](auto& cueData) {
        return VTTCue::create(protectedDocument(), cueData);
    });
}

Vector<Ref<VTTRegion>> TextTrackLoader::getNewRegions()
{
    ASSERT(m_cueParser);
    if (!m_cueParser)
        return { };
    return m_cueParser->takeRegions();
}

Vector<String> TextTrackLoader::getNewStyleSheets()
{
    ASSERT(m_cueParser);
    if (!m_cueParser)
        return { };
    return m_cueParser->takeStyleSheets();
}

Ref<Document> TextTrackLoader::protectedDocument() const
{
    return m_document.get();
}

CachedResourceHandle<CachedTextTrack> TextTrackLoader::protectedResource() const
{
    return m_resource.get();
}

} // namespace WebCore

#endif
