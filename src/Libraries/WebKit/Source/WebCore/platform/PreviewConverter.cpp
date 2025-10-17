/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 8, 2025.
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
#include "PreviewConverter.h"

#if ENABLE(PREVIEW_CONVERTER)

#include "PreviewConverterClient.h"
#include "PreviewConverterProvider.h"
#include <wtf/RunLoop.h>
#include <wtf/SetForScope.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(PreviewConverter);

PreviewConverter::~PreviewConverter() = default;

bool PreviewConverter::supportsMIMEType(const String& mimeType)
{
    if (mimeType.isNull())
        return false;

    if (equalLettersIgnoringASCIICase(mimeType, "text/html"_s) || equalLettersIgnoringASCIICase(mimeType, "text/plain"_s))
        return false;

    static NeverDestroyed<UncheckedKeyHashSet<String, ASCIICaseInsensitiveHash>> supportedMIMETypes = platformSupportedMIMETypes();
    return supportedMIMETypes->contains(mimeType);
}

ResourceResponse PreviewConverter::previewResponse() const
{
    auto response = platformPreviewResponse();
    ASSERT(response.mimeType().length());
    response.setIsQuickLook(true);
    return response;
}

const ResourceError& PreviewConverter::previewError() const
{
    return m_previewError;
}

const FragmentedSharedBuffer& PreviewConverter::previewData() const
{
    return *m_previewData.get();
}

void PreviewConverter::updateMainResource()
{
    if (m_isInClientCallback)
        return;

    if (m_state != State::Updating)
        return;

    auto provider = m_provider.get();
    if (!provider) {
        didFailUpdating();
        return;
    }

    provider->provideMainResourceForPreviewConverter(*this, [this, protectedThis = Ref { *this }](Ref<FragmentedSharedBuffer>&& buffer) {
        appendFromBuffer(WTFMove(buffer));
    });
}

void PreviewConverter::appendFromBuffer(const FragmentedSharedBuffer& buffer)
{
    while (buffer.size() > m_lengthAppended) {
        auto newData = buffer.getSomeData(m_lengthAppended);
        platformAppend(newData);
        m_lengthAppended += newData.size();
    }
}

void PreviewConverter::finishUpdating()
{
    if (m_isInClientCallback)
        return;

    if (m_state != State::Updating)
        return;

    platformFinishedAppending();

    iterateClients([&](auto& client) {
        client.previewConverterDidFinishUpdating(*this);
    });
}

void PreviewConverter::failedUpdating()
{
    if (m_isInClientCallback)
        return;

    if (m_state != State::Updating)
        return;

    m_state = State::FailedUpdating;
    platformFailedAppending();
}

bool PreviewConverter::hasClient(PreviewConverterClient& client) const
{
    return m_clients.contains(&client);
}

void PreviewConverter::addClient(PreviewConverterClient& client)
{
    ASSERT(!hasClient(client));
    m_clients.append(client);
    didAddClient(client);
}

void PreviewConverter::removeClient(PreviewConverterClient& client)
{
    m_clients.removeFirst(&client);
    ASSERT(!hasClient(client));
}

static String& sharedPasswordForTesting()
{
    static NeverDestroyed<String> passwordForTesting;
    return passwordForTesting.get();
}

const String& PreviewConverter::passwordForTesting()
{
    return sharedPasswordForTesting();
}

void PreviewConverter::setPasswordForTesting(const String& password)
{
    sharedPasswordForTesting() = password;
}

template<typename T>
void PreviewConverter::iterateClients(T&& callback)
{
    SetForScope isInClientCallback { m_isInClientCallback, true };
    auto clientsCopy { m_clients };
    auto protectedThis { Ref { *this } };

    for (auto& client : clientsCopy) {
        if (client && hasClient(*client))
            callback(*client);
    }
}

void PreviewConverter::didAddClient(PreviewConverterClient& client)
{
    RunLoop::current().dispatch([this, protectedThis = Ref { *this }, weakClient = WeakPtr { client }]() {
        if (auto client = weakClient.get())
            replayToClient(*client);
    });
}

void PreviewConverter::didFailConvertingWithError(const ResourceError& error)
{
    m_previewError = error;
    m_state = State::FailedConverting;

    iterateClients([&](auto& client) {
        client.previewConverterDidFailConverting(*this);
    });
}

void PreviewConverter::didFailUpdating()
{
    failedUpdating();

    iterateClients([&](auto& client) {
        client.previewConverterDidFailUpdating(*this);
    });
}

void PreviewConverter::replayToClient(PreviewConverterClient& client)
{
    if (!hasClient(client))
        return;

    SetForScope isInClientCallback { m_isInClientCallback, true };
    auto protectedThis { Ref { *this } };

    client.previewConverterDidStartUpdating(*this);

    if (m_state == State::Updating || !hasClient(client))
        return;

    if (m_state == State::FailedUpdating) {
        client.previewConverterDidFailUpdating(*this);
        return;
    }

    ASSERT(m_state >= State::Converting);
    client.previewConverterDidStartConverting(*this);

    if (!m_previewData.isEmpty() && hasClient(client))
        client.previewConverterDidReceiveData(*this, *m_previewData.get());

    if (m_state == State::Converting || !hasClient(client))
        return;

    if (m_state == State::FailedConverting) {
        ASSERT(!m_previewError.isNull());
        client.previewConverterDidFailConverting(*this);
        return;
    }

    ASSERT(m_state == State::FinishedConverting);
    ASSERT(!m_previewData.isEmpty());
    ASSERT(m_previewError.isNull());
    client.previewConverterDidFinishConverting(*this);
}

void PreviewConverter::delegateDidReceiveData(const FragmentedSharedBuffer& data)
{
    auto protectedThis { Ref { *this } };

    if (m_state == State::Updating) {
        m_provider = nullptr;
        m_state = State::Converting;

        iterateClients([&](auto& client) {
            client.previewConverterDidStartConverting(*this);
        });
    }

    ASSERT(m_state == State::Converting);
    if (data.isEmpty())
        return;

    m_previewData.append(data);

    iterateClients([&](auto& client) {
        client.previewConverterDidReceiveData(*this, data);
    });
}

void PreviewConverter::delegateDidFinishLoading()
{
    ASSERT(m_state == State::Converting);
    m_state = State::FinishedConverting;

    iterateClients([&](auto& client) {
        client.previewConverterDidFinishConverting(*this);
    });
}

void PreviewConverter::delegateDidFailWithError(const ResourceError& error)
{
    if (!isPlatformPasswordError(error)) {
        didFailConvertingWithError(error);
        return;
    }

    ASSERT(m_state == State::Updating);
    auto provider = m_provider.get();
    if (!provider) {
        didFailConvertingWithError(error);
        return;
    }

    provider->providePasswordForPreviewConverter(*this, [this, protectedThis = Ref { *this }](auto& password) mutable {
        if (m_state != State::Updating)
            return;

        platformUnlockWithPassword(password);
        m_lengthAppended = 0;
        updateMainResource();
        finishUpdating();
    });
}

} // namespace WebCore

#endif // ENABLE(PREVIEW_CONVERTER)
