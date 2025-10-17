/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 9, 2023.
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

#if USE(LIBWEBRTC)

#include "LibWebRTCMacros.h"
#include "ScriptExecutionContextIdentifier.h"
#include "WebRTCProvider.h"
#include <wtf/Compiler.h>
#include <wtf/TZoneMalloc.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN

// See Bug 274508: Disable thread-safety-reference-return warnings in libwebrtc
IGNORE_CLANG_WARNINGS_BEGIN("thread-safety-reference-return")
#include <webrtc/api/peer_connection_interface.h>
IGNORE_CLANG_WARNINGS_END
#include <webrtc/api/scoped_refptr.h>
#include <webrtc/api/video_codecs/video_decoder_factory.h>
#include <webrtc/api/video_codecs/video_encoder_factory.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

namespace rtc {
class NetworkManager;
class PacketSocketFactory;
class Thread;
class RTCCertificateGenerator;
}

namespace webrtc {
class AsyncDnsResolverFactory;
class PeerConnectionFactoryInterface;
}

namespace WebCore {

class LibWebRTCAudioModule;
class RegistrableDomain;
struct PeerConnectionFactoryAndThreads;

class WEBCORE_EXPORT LibWebRTCProvider : public WebRTCProvider {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(LibWebRTCProvider, WEBCORE_EXPORT);
public:
    static UniqueRef<LibWebRTCProvider> create();

    virtual ~LibWebRTCProvider();

    static void registerWebKitVP9Decoder();

    static void setRTCLogging(WTFLogLevel);

    virtual void setEnableWebRTCEncryption(bool);
    virtual void disableNonLocalhostConnections();

    std::optional<RTCRtpCapabilities> receiverCapabilities(const String& kind) final;
    std::optional<RTCRtpCapabilities> senderCapabilities(const String& kind) final;
    void clearFactory() final;

    virtual rtc::scoped_refptr<webrtc::PeerConnectionInterface> createPeerConnection(ScriptExecutionContextIdentifier, webrtc::PeerConnectionObserver&, rtc::PacketSocketFactory*, webrtc::PeerConnectionInterface::RTCConfiguration&&);

    webrtc::PeerConnectionFactoryInterface* factory();
    LibWebRTCAudioModule* audioModule();

    // FIXME: Make these methods not static.
    static void callOnWebRTCNetworkThread(Function<void()>&&);
    static void callOnWebRTCSignalingThread(Function<void()>&&);
    static bool hasWebRTCThreads();
    static rtc::Thread& signalingThread();

    // Used for mock testing
    void setPeerConnectionFactory(rtc::scoped_refptr<webrtc::PeerConnectionFactoryInterface>&&);

    // Callback is executed on a background thread.
    void prepareCertificateGenerator(Function<void(rtc::RTCCertificateGenerator&)>&&);

    virtual void setLoggingLevel(WTFLogLevel);

    WEBCORE_EXPORT virtual void setVP9HardwareSupportForTesting(std::optional<bool> value) { m_supportsVP9VTBForTesting = value; }
    virtual bool isSupportingVP9HardwareDecoder() const { return m_supportsVP9VTBForTesting.value_or(false); }

    WEBCORE_EXPORT void disableEnumeratingAllNetworkInterfaces();
    WEBCORE_EXPORT void enableEnumeratingAllNetworkInterfaces();
    bool isEnumeratingAllNetworkInterfacesEnabled() const;
    WEBCORE_EXPORT void enableEnumeratingVisibleNetworkInterfaces();
    bool isEnumeratingVisibleNetworkInterfacesEnabled() const { return m_enableEnumeratingVisibleNetworkInterfaces; }

    class SuspendableSocketFactory : public rtc::PacketSocketFactory {
    public:
        virtual ~SuspendableSocketFactory() = default;
        virtual void suspend() { };
        virtual void resume() { };
        virtual void disableRelay() { };
    };
    virtual std::unique_ptr<SuspendableSocketFactory> createSocketFactory(String&& /* userAgent */, ScriptExecutionContextIdentifier, bool /* isFirstParty */, RegistrableDomain&&);

protected:
    LibWebRTCProvider();

    rtc::scoped_refptr<webrtc::PeerConnectionInterface> createPeerConnection(webrtc::PeerConnectionObserver&, rtc::NetworkManager&, rtc::PacketSocketFactory&, webrtc::PeerConnectionInterface::RTCConfiguration&&, std::unique_ptr<webrtc::AsyncDnsResolverFactoryInterface>&&);

    rtc::scoped_refptr<webrtc::PeerConnectionFactoryInterface> createPeerConnectionFactory(rtc::Thread* networkThread, rtc::Thread* signalingThread);
    virtual std::unique_ptr<webrtc::VideoDecoderFactory> createDecoderFactory();
    virtual std::unique_ptr<webrtc::VideoEncoderFactory> createEncoderFactory();

    virtual void startedNetworkThread();

    PeerConnectionFactoryAndThreads& getStaticFactoryAndThreads(bool useNetworkThreadWithSocketServer);

    RefPtr<LibWebRTCAudioModule> m_audioModule;
    rtc::scoped_refptr<webrtc::PeerConnectionFactoryInterface> m_factory;
    // FIXME: Remove m_useNetworkThreadWithSocketServer member variable and make it a global.
    bool m_useNetworkThreadWithSocketServer { true };

private:
    void initializeAudioDecodingCapabilities() final;
    void initializeVideoDecodingCapabilities() final;
    void initializeAudioEncodingCapabilities() final;
    void initializeVideoEncodingCapabilities() final;

    virtual void willCreatePeerConnectionFactory();

    std::optional<MediaCapabilitiesDecodingInfo> videoDecodingCapabilitiesOverride(const VideoConfiguration&) final;
    std::optional<MediaCapabilitiesEncodingInfo> videoEncodingCapabilitiesOverride(const VideoConfiguration&) final;

    std::optional<bool> m_supportsVP9VTBForTesting { false };
    bool m_disableNonLocalhostConnections { false };
    bool m_enableEnumeratingAllNetworkInterfaces { false };
    bool m_enableEnumeratingVisibleNetworkInterfaces { false };
};

inline void LibWebRTCProvider::willCreatePeerConnectionFactory()
{
}

inline LibWebRTCAudioModule* LibWebRTCProvider::audioModule()
{
    return m_audioModule.get();
}

} // namespace WebCore

#endif // USE(LIBWEBRTC)
