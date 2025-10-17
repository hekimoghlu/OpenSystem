/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 31, 2024.
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
#include "APIAttachment.h"

#if ENABLE(ATTACHMENT_ELEMENT)

#include "WebPageProxy.h"
#include <WebCore/SharedBuffer.h>
#include <wtf/CompletionHandler.h>
#include <wtf/text/WTFString.h>

namespace API {

Ref<Attachment> Attachment::create(const WTF::String& identifier, WebKit::WebPageProxy& webPage)
{
    return adoptRef(*new Attachment(identifier, webPage));
}

Attachment::Attachment(const WTF::String& identifier, WebKit::WebPageProxy& webPage)
    : m_identifier(identifier)
    , m_webPage(webPage)
{
}

Attachment::~Attachment()
{
}

void Attachment::updateAttributes(CompletionHandler<void()>&& callback)
{
    if (!m_webPage) {
        callback();
        return;
    }

    if (m_webPage->willUpdateAttachmentAttributes(*this) == WebKit::WebPageProxy::ShouldUpdateAttachmentAttributes::No) {
        callback();
        return;
    }

    m_webPage->updateAttachmentAttributes(*this, WTFMove(callback));
}

void Attachment::invalidate()
{
    m_identifier = { };
    m_filePath = { };
    m_contentType = { };
    m_webPage.clear();
    m_insertionState = InsertionState::NotInserted;
#if PLATFORM(COCOA)
    Locker locker { m_fileWrapperLock };
    m_fileWrapper.clear();
#endif
}

#if !PLATFORM(COCOA)

bool Attachment::isEmpty() const
{
    return true;
}

WTF::String Attachment::mimeType() const
{
    return m_contentType;
}

WTF::String Attachment::fileName() const
{
    return { };
}

std::optional<uint64_t> Attachment::fileSizeForDisplay() const
{
    return std::nullopt;
}

RefPtr<WebCore::FragmentedSharedBuffer> Attachment::associatedElementData() const
{
    return nullptr;
}

RefPtr<WebCore::SharedBuffer> Attachment::createSerializedRepresentation() const
{
    return nullptr;
}

void Attachment::updateFromSerializedRepresentation(Ref<WebCore::SharedBuffer>&&, const WTF::String&)
{
}

#endif // !PLATFORM(COCOA)

}

#endif // ENABLE(ATTACHMENT_ELEMENT)
