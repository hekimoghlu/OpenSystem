/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 25, 2024.
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
#pragma once

#if ENABLE(GPU_PROCESS)

#include "AudioMediaStreamTrackRendererInternalUnitIdentifier.h"
#include "Connection.h"
#include "GPUProcessConnectionIdentifier.h"
#include "GraphicsContextGLIdentifier.h"
#include "MediaOverridesForTesting.h"
#include "MessageReceiverMap.h"
#include "RenderingBackendIdentifier.h"
#include "StreamServerConnection.h"
#include "WebGPUIdentifier.h"
#include <WebCore/AudioSession.h>
#include <WebCore/PlatformMediaSession.h>
#include <WebCore/SharedMemory.h>
#include <wtf/AbstractThreadSafeRefCountedAndCanMakeWeakPtr.h>
#include <wtf/Forward.h>
#include <wtf/RefCounted.h>
#include <wtf/ThreadSafeWeakHashSet.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class CAAudioStreamDescription;
struct GraphicsContextGLAttributes;
struct PageIdentifierType;
using PageIdentifier = ObjectIdentifier<PageIdentifierType>;
}

namespace IPC {
class Semaphore;
}

namespace WebKit {
class RemoteAudioSourceProviderManager;
class RemoteMediaPlayerManager;
class RemoteSharedResourceCacheProxy;
class SampleBufferDisplayLayerManager;
class WebPage;
struct GPUProcessConnectionInfo;
struct OverrideScreenDataForTesting;
struct WebPageCreationParameters;

#if ENABLE(VIDEO)
class RemoteVideoFrameObjectHeapProxy;
#endif

class GPUProcessConnection : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<GPUProcessConnection>, public IPC::Connection::Client {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(GPUProcessConnection);
public:
    static Ref<GPUProcessConnection> create(Ref<IPC::Connection>&&);
    ~GPUProcessConnection();
    GPUProcessConnectionIdentifier identifier() const { return m_identifier; }

    void ref() const final { ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr::ref(); }
    void deref() const final { ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr::deref(); }

    IPC::Connection& connection() { return m_connection.get(); }
    Ref<IPC::Connection> protectedConnection() { return m_connection; }
    IPC::MessageReceiverMap& messageReceiverMap() { return m_messageReceiverMap; }

    void didBecomeUnresponsive();
#if HAVE(AUDIT_TOKEN)
    std::optional<audit_token_t> auditToken();
#endif
    Ref<RemoteSharedResourceCacheProxy> sharedResourceCache();
#if PLATFORM(COCOA) && ENABLE(MEDIA_STREAM)
    SampleBufferDisplayLayerManager& sampleBufferDisplayLayerManager();
    Ref<SampleBufferDisplayLayerManager> protectedSampleBufferDisplayLayerManager();
    void resetAudioMediaStreamTrackRendererInternalUnit(AudioMediaStreamTrackRendererInternalUnitIdentifier);
#endif
#if ENABLE(VIDEO)
    RemoteVideoFrameObjectHeapProxy& videoFrameObjectHeapProxy();
    Ref<RemoteVideoFrameObjectHeapProxy> protectedVideoFrameObjectHeapProxy();
    RemoteMediaPlayerManager& mediaPlayerManager();
#endif

#if PLATFORM(COCOA) && ENABLE(WEB_AUDIO)
    RemoteAudioSourceProviderManager& audioSourceProviderManager();
#endif

    void updateMediaConfiguration(bool forceUpdate);

#if HAVE(VISIBILITY_PROPAGATION_VIEW)
    void createVisibilityPropagationContextForPage(WebPage&);
    void destroyVisibilityPropagationContextForPage(WebPage&);
#endif

#if ENABLE(EXTENSION_CAPABILITIES)
    void setMediaEnvironment(WebCore::PageIdentifier, const String&);
#endif

    void configureLoggingChannel(const String&, WTFLogChannelState, WTFLogLevel);

    void createRenderingBackend(RenderingBackendIdentifier, IPC::StreamServerConnection::Handle&&);
    void releaseRenderingBackend(RenderingBackendIdentifier);
#if ENABLE(WEBGL)
    void createGraphicsContextGL(GraphicsContextGLIdentifier, const WebCore::GraphicsContextGLAttributes&, RenderingBackendIdentifier, IPC::StreamServerConnection::Handle&&);
    void releaseGraphicsContextGL(GraphicsContextGLIdentifier);
#endif
    void createGPU(WebGPUIdentifier, RenderingBackendIdentifier, IPC::StreamServerConnection::Handle&&);
    void releaseGPU(WebGPUIdentifier);

    class Client : public AbstractThreadSafeRefCountedAndCanMakeWeakPtr {
    public:
        virtual ~Client() = default;

        virtual void gpuProcessConnectionDidClose(GPUProcessConnection&) { }
    };
    void addClient(const Client& client) { m_clients.add(client); }

    static constexpr Seconds defaultTimeout = 3_s;
private:
    GPUProcessConnection(Ref<IPC::Connection>&&);
    bool waitForDidInitialize();
    void invalidate();

    // IPC::Connection::Client
    void didClose(IPC::Connection&) override;
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;
    bool didReceiveSyncMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&) final;
    void didReceiveInvalidMessage(IPC::Connection&, IPC::MessageName, int32_t indexOfObjectFailingDecoding) override;

    bool dispatchMessage(IPC::Connection&, IPC::Decoder&);
    bool dispatchSyncMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&);

    // Messages.
    void didReceiveRemoteCommand(WebCore::PlatformMediaSession::RemoteControlCommandType, const WebCore::PlatformMediaSession::RemoteCommandArgument&);
    void didInitialize(std::optional<GPUProcessConnectionInfo>&&);

#if ENABLE(ROUTING_ARBITRATION)
    void beginRoutingArbitrationWithCategory(WebCore::AudioSession::CategoryType, WebCore::AudioSessionRoutingArbitrationClient::ArbitrationCallback&&);
    void endRoutingArbitration();
#endif

    // The connection from the web process to the GPU process.
    Ref<IPC::Connection> m_connection;
    IPC::MessageReceiverMap m_messageReceiverMap;
    GPUProcessConnectionIdentifier m_identifier { GPUProcessConnectionIdentifier::generate() };
    bool m_hasInitialized { false };
    RefPtr<RemoteSharedResourceCacheProxy> m_sharedResourceCache;
#if HAVE(AUDIT_TOKEN)
    std::optional<audit_token_t> m_auditToken;
#endif
#if PLATFORM(COCOA) && ENABLE(MEDIA_STREAM)
    std::unique_ptr<SampleBufferDisplayLayerManager> m_sampleBufferDisplayLayerManager;
#endif
#if ENABLE(VIDEO)
    RefPtr<RemoteVideoFrameObjectHeapProxy> m_videoFrameObjectHeapProxy;
#endif
#if PLATFORM(COCOA) && ENABLE(WEB_AUDIO)
    RefPtr<RemoteAudioSourceProviderManager> m_audioSourceProviderManager;
#endif

#if PLATFORM(COCOA)
    MediaOverridesForTesting m_mediaOverridesForTesting;
#endif

    ThreadSafeWeakHashSet<Client> m_clients;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
