/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 13, 2022.
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
#include "RemoteRealtimeMediaSourceProxy.h"

#if PLATFORM(COCOA) && ENABLE(MEDIA_STREAM)

#include "GPUProcessConnection.h"
#include "SharedCARingBuffer.h"
#include "UserMediaCaptureManager.h"
#include "UserMediaCaptureManagerMessages.h"
#include "UserMediaCaptureManagerProxyMessages.h"
#include "WebProcess.h"
#include <WebCore/MediaConstraints.h>
#include <WebCore/RealtimeMediaSource.h>
#include <WebCore/RealtimeMediaSourceCenter.h>
#include <WebCore/WebAudioBufferList.h>
#include <wtf/NativePromise.h>

namespace WebKit {
using namespace WebCore;

static Ref<IPC::Connection> getSourceConnection(bool shouldCaptureInGPUProcess)
{
    ASSERT(isMainRunLoop());
#if ENABLE(GPU_PROCESS)
    if (shouldCaptureInGPUProcess)
        return WebProcess::singleton().ensureGPUProcessConnection().connection();
#endif
    return *WebProcess::singleton().parentProcessConnection();
}

RemoteRealtimeMediaSourceProxy::RemoteRealtimeMediaSourceProxy(WebCore::RealtimeMediaSourceIdentifier identifier, const WebCore::CaptureDevice& device, bool shouldCaptureInGPUProcess, const WebCore::MediaConstraints* constraints)
    : m_identifier(identifier)
    , m_connection(getSourceConnection(shouldCaptureInGPUProcess))
    , m_device(device)
    , m_shouldCaptureInGPUProcess(shouldCaptureInGPUProcess)
{
    if (constraints)
        m_constraints = *constraints;
}

RemoteRealtimeMediaSourceProxy::~RemoteRealtimeMediaSourceProxy()
{
    failApplyConstraintCallbacks("Source terminated"_s);
}

void RemoteRealtimeMediaSourceProxy::updateConnection()
{
    m_connection = getSourceConnection(m_shouldCaptureInGPUProcess);
}

void RemoteRealtimeMediaSourceProxy::startProducingData(WebCore::PageIdentifier pageIdentifier)
{
    m_interrupted = false;
    m_connection->send(Messages::UserMediaCaptureManagerProxy::StartProducingData { m_identifier, pageIdentifier }, 0);
}

void RemoteRealtimeMediaSourceProxy::stopProducingData()
{
    m_interrupted = false;
    m_connection->send(Messages::UserMediaCaptureManagerProxy::StopProducingData { m_identifier }, 0);
}

void RemoteRealtimeMediaSourceProxy::endProducingData()
{
    m_connection->send(Messages::UserMediaCaptureManagerProxy::EndProducingData { m_identifier }, 0);
}

void RemoteRealtimeMediaSourceProxy::createRemoteMediaSource(const MediaDeviceHashSalts& deviceIDHashSalts, WebCore::PageIdentifier pageIdentifier, CreateCallback&& callback, bool shouldUseRemoteFrame)
{
    m_connection->sendWithAsyncReply(Messages::UserMediaCaptureManagerProxy::CreateMediaSourceForCaptureDeviceWithConstraints(identifier(), m_device, deviceIDHashSalts, m_constraints, shouldUseRemoteFrame, pageIdentifier), WTFMove(callback));
}

RemoteRealtimeMediaSourceProxy RemoteRealtimeMediaSourceProxy::clone()
{
    RemoteRealtimeMediaSourceProxy clone = { RealtimeMediaSourceIdentifier::generate(), m_device, m_shouldCaptureInGPUProcess, &m_constraints };

    clone.m_interrupted = m_interrupted;
    clone.m_isEnded = m_isEnded;

    return clone;
}

void RemoteRealtimeMediaSourceProxy::createRemoteCloneSource(WebCore::RealtimeMediaSourceIdentifier cloneIdentifier, WebCore::PageIdentifier pageIdentifier)
{
    m_connection->send(Messages::UserMediaCaptureManagerProxy::Clone { m_identifier, cloneIdentifier, pageIdentifier }, 0);
}

void RemoteRealtimeMediaSourceProxy::applyConstraints(const MediaConstraints& constraints, RealtimeMediaSource::ApplyConstraintsHandler&& completionHandler)
{
    m_pendingApplyConstraintsRequests.append(std::make_pair(WTFMove(completionHandler), constraints));
    // FIXME: Use sendAsyncWithReply.
    m_connection->send(Messages::UserMediaCaptureManagerProxy::ApplyConstraints { m_identifier, constraints }, 0);
}

struct RemoteRealtimeMediaSourceProxy::PromiseConverter {
    static auto convertError(IPC::Error)
    {
        return makeUnexpected(String { "IPC Connection closed"_s });
    }
};

Ref<WebCore::RealtimeMediaSource::TakePhotoNativePromise> RemoteRealtimeMediaSourceProxy::takePhoto(PhotoSettings&& settings)
{
    return m_connection->sendWithPromisedReply<PromiseConverter>(Messages::UserMediaCaptureManagerProxy::TakePhoto { identifier(), WTFMove(settings) });
}

Ref<WebCore::RealtimeMediaSource::PhotoCapabilitiesNativePromise> RemoteRealtimeMediaSourceProxy::getPhotoCapabilities()
{
    return m_connection->sendWithPromisedReply<PromiseConverter>(Messages::UserMediaCaptureManagerProxy::GetPhotoCapabilities { identifier() });
}

Ref<WebCore::RealtimeMediaSource::PhotoSettingsNativePromise> RemoteRealtimeMediaSourceProxy::getPhotoSettings()
{
    return m_connection->sendWithPromisedReply<PromiseConverter>(Messages::UserMediaCaptureManagerProxy::GetPhotoSettings { identifier() });
}

void RemoteRealtimeMediaSourceProxy::applyConstraintsSucceeded()
{
    auto request = m_pendingApplyConstraintsRequests.takeFirst();
    m_constraints = WTFMove(request.second);
    request.first({ });
}

void RemoteRealtimeMediaSourceProxy::applyConstraintsFailed(WebCore::MediaConstraintType invalidConstraint, String&& errorMessage)
{
    auto callback = m_pendingApplyConstraintsRequests.takeFirst().first;
    callback(RealtimeMediaSource::ApplyConstraintsError { invalidConstraint, WTFMove(errorMessage) });
}

void RemoteRealtimeMediaSourceProxy::failApplyConstraintCallbacks(const String& errorMessage)
{
    auto requests = WTFMove(m_pendingApplyConstraintsRequests);
    while (!requests.isEmpty())
        requests.takeFirst().first(RealtimeMediaSource::ApplyConstraintsError { { }, errorMessage });
}

void RemoteRealtimeMediaSourceProxy::end()
{
    ASSERT(!m_isEnded);
    m_isEnded = true;
    m_connection->send(Messages::UserMediaCaptureManagerProxy::RemoveSource { m_identifier }, 0);
}

void RemoteRealtimeMediaSourceProxy::whenReady(CompletionHandler<void(WebCore::CaptureSourceError&&)>&& callback)
{
    if (m_isReady)
        return callback(WebCore::CaptureSourceError(m_failureReason));

    if (m_callback) {
        callback = [previousCallbacks = std::exchange(m_callback, { }), newCallback = WTFMove(callback)] (auto&& error) mutable {
            previousCallbacks(WebCore::CaptureSourceError { error });
            newCallback(WTFMove(error));
        };
    }

    m_callback = WTFMove(callback);
}

void RemoteRealtimeMediaSourceProxy::setAsReady()
{
    ASSERT(!m_isReady);
    m_isReady = true;
    if (m_callback)
        m_callback({ });
}

void RemoteRealtimeMediaSourceProxy::didFail(CaptureSourceError&& reason)
{
    m_isReady = true;
    m_failureReason = WTFMove(reason);
    if (m_callback)
        m_callback(WebCore::CaptureSourceError(m_failureReason));
}

bool RemoteRealtimeMediaSourceProxy::isPowerEfficient() const
{
    auto syncResult = m_connection->sendSync(Messages::UserMediaCaptureManagerProxy::IsPowerEfficient { identifier() }, 0, GPUProcessConnection::defaultTimeout);
    auto [isPowerEfficient] = syncResult.takeReplyOr(false);
    return isPowerEfficient;
}

}

#endif
