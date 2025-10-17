/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 13, 2022.
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
#include "SocketStreamHandleImpl.h"

#if PLATFORM(COCOA)

#include "SocketStreamHandleClient.h"
#include <WebCore/CookieRequestHeaderFieldProxy.h>
#include <WebCore/NetworkStorageSession.h>
#include <WebCore/StorageSessionProvider.h>
#include <wtf/Function.h>

namespace WebCore {

void SocketStreamHandleImpl::platformSend(std::span<const uint8_t> data, Function<void(bool)>&& completionHandler)
{
    if (!m_buffer.isEmpty()) {
        if (m_buffer.size() + data.size() > maxBufferSize) {
            // FIXME: report error to indicate that buffer has no more space.
            return completionHandler(false);
        }
        m_buffer.append(data);
        m_client.didUpdateBufferedAmount(*this, bufferedAmount());
        return completionHandler(true);
    }
    size_t bytesWritten = 0;
    if (m_state == Open) {
        if (auto result = platformSendInternal(data))
            bytesWritten = result.value();
        else
            return completionHandler(false);
    }
    if (m_buffer.size() + data.size() - bytesWritten > maxBufferSize) {
        // FIXME: report error to indicate that buffer has no more space.
        return completionHandler(false);
    }
    if (bytesWritten < data.size()) {
        m_buffer.append(data.subspan(bytesWritten));
        m_client.didUpdateBufferedAmount(static_cast<SocketStreamHandle&>(*this), bufferedAmount());
    }
    return completionHandler(true);
}

static std::span<const uint8_t> removeTerminationCharacters(std::span<const uint8_t> data)
{
#ifndef NDEBUG
    ASSERT(data.size() > 2);
    ASSERT(data[data.size() - 2] == '\r');
    ASSERT(data[data.size() - 1] == '\n');
#else
    UNUSED_PARAM(data);
#endif

    // Remove the terminating '\r\n'
    return data.first(data.size() - 2);
}

static std::optional<std::pair<Vector<uint8_t>, bool>> cookieDataForHandshake(const NetworkStorageSession* networkStorageSession, const CookieRequestHeaderFieldProxy& headerFieldProxy)
{
    if (!networkStorageSession)
        return std::nullopt;
    
    auto [cookieDataString, secureCookiesAccessed] = networkStorageSession->cookieRequestHeaderFieldValue(headerFieldProxy);
    if (cookieDataString.isEmpty())
        return std::pair<Vector<uint8_t>, bool> { { }, secureCookiesAccessed };

    Vector<uint8_t> data = { 'C', 'o', 'o', 'k', 'i', 'e', ':', ' ' };
    data.append(cookieDataString.utf8().span());
    data.append("\r\n\r\n"_span);

    return std::pair<Vector<uint8_t>, bool> { data, secureCookiesAccessed };
}

void SocketStreamHandleImpl::platformSendHandshake(std::span<const uint8_t> data, const std::optional<CookieRequestHeaderFieldProxy>& headerFieldProxy, Function<void(bool, bool)>&& completionHandler)
{
    Vector<uint8_t> cookieData;
    bool secureCookiesAccessed = false;

    if (headerFieldProxy) {
        auto cookieDataFromNetworkSession = cookieDataForHandshake(m_storageSessionProvider ? m_storageSessionProvider->storageSession() : nullptr, *headerFieldProxy);
        if (!cookieDataFromNetworkSession) {
            completionHandler(false, false);
            return;
        }

        std::tie(cookieData, secureCookiesAccessed) = *cookieDataFromNetworkSession;
        if (cookieData.size())
            data = removeTerminationCharacters(data);
    }

    if (!m_buffer.isEmpty()) {
        if (m_buffer.size() + data.size() + cookieData.size() > maxBufferSize) {
            // FIXME: report error to indicate that buffer has no more space.
            return completionHandler(false, secureCookiesAccessed);
        }
        m_buffer.append(data);
        m_buffer.append(cookieData.data(), cookieData.size());
        m_client.didUpdateBufferedAmount(*this, bufferedAmount());
        return completionHandler(true, secureCookiesAccessed);
    }
    size_t bytesWritten = 0;
    if (m_state == Open) {
        // Unfortunately, we need to send the data in one buffer or else the handshake fails.
        Vector<uint8_t> sendData;
        sendData.reserveInitialCapacity(data.size() + cookieData.size());
        sendData.append(data);
        sendData.appendVector(cookieData);

        if (auto result = platformSendInternal(sendData.span()))
            bytesWritten = result.value();
        else
            return completionHandler(false, secureCookiesAccessed);
    }
    if (m_buffer.size() + data.size() + cookieData.size() - bytesWritten > maxBufferSize) {
        // FIXME: report error to indicate that buffer has no more space.
        return completionHandler(false, secureCookiesAccessed);
    }
    if (bytesWritten < data.size() + cookieData.size()) {
        size_t cookieBytesWritten = 0;
        if (bytesWritten < data.size())
            m_buffer.append(data.subspan(bytesWritten));
        else
            cookieBytesWritten = bytesWritten - data.size();
        m_buffer.append(cookieData.data() + cookieBytesWritten, cookieData.size() - cookieBytesWritten);
        m_client.didUpdateBufferedAmount(static_cast<SocketStreamHandle&>(*this), bufferedAmount());
    }
    return completionHandler(true, secureCookiesAccessed);
}

bool SocketStreamHandleImpl::sendPendingData()
{
    if (m_state != Open && m_state != Closing)
        return false;
    if (m_buffer.isEmpty()) {
        if (m_state == Open)
            return false;
        if (m_state == Closing) {
            disconnect();
            return false;
        }
    }
    bool pending;
    do {
        auto result = platformSendInternal(m_buffer.firstBlockSpan());
        if (!result)
            return false;
        size_t bytesWritten = result.value();
        if (!bytesWritten)
            return false;
        pending = bytesWritten != m_buffer.firstBlockSize();
        ASSERT(m_buffer.size() - bytesWritten <= maxBufferSize);
        m_buffer.consume(bytesWritten);
    } while (!pending && !m_buffer.isEmpty());
    m_client.didUpdateBufferedAmount(static_cast<SocketStreamHandle&>(*this), bufferedAmount());
    return true;
}

size_t SocketStreamHandleImpl::bufferedAmount()
{
    return m_buffer.size();
}

} // namespace WebCore

#endif // PLATFORM(COCOA)
