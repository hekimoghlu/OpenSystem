/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 24, 2023.
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

#if ENABLE(WEB_RTC) && USE(LIBWEBRTC)

#include "ExceptionCode.h"
#include "LibWebRTCMacros.h"
#include "LibWebRTCUtils.h"

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN

// See Bug 274508: Disable thread-safety-reference-return warnings in libwebrtc
IGNORE_CLANG_WARNINGS_BEGIN("thread-safety-reference-return")
#include <webrtc/api/peer_connection_interface.h>
IGNORE_CLANG_WARNINGS_END

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

namespace WebCore {

template<typename Endpoint>
class CreateSessionDescriptionObserver final : public webrtc::CreateSessionDescriptionObserver {
public:
    explicit CreateSessionDescriptionObserver(Endpoint &endpoint)
        : m_endpoint(endpoint)
    {
    }

    void OnSuccess(webrtc::SessionDescriptionInterface* sessionDescription) final { m_endpoint.createSessionDescriptionSucceeded(std::unique_ptr<webrtc::SessionDescriptionInterface>(sessionDescription)); }
    void OnFailure(webrtc::RTCError error) final { m_endpoint.createSessionDescriptionFailed(toExceptionCode(error.type()), error.message()); }

    void AddRef() const { m_endpoint.AddRef(); }
    webrtc::RefCountReleaseStatus Release() const { return m_endpoint.Release(); }

private:
    Endpoint& m_endpoint;
};

template<typename Endpoint>
class SetLocalSessionDescriptionObserver final : public webrtc::SetLocalDescriptionObserverInterface {
public:
    explicit SetLocalSessionDescriptionObserver(Endpoint &endpoint)
        : m_endpoint(endpoint)
    {
    }

    void AddRef() const { m_endpoint.AddRef(); }
    webrtc::RefCountReleaseStatus Release() const { return m_endpoint.Release(); }

private:
    void OnSetLocalDescriptionComplete(webrtc::RTCError error) final
    {
        if (!error.ok()) {
            m_endpoint.setLocalSessionDescriptionFailed(toExceptionCode(error.type()), error.message());
            return;
        }
        m_endpoint.setLocalSessionDescriptionSucceeded();
    }

    Endpoint& m_endpoint;
};

template<typename Endpoint>
class SetRemoteSessionDescriptionObserver final : public webrtc::SetRemoteDescriptionObserverInterface {
public:
    explicit SetRemoteSessionDescriptionObserver(Endpoint &endpoint)
        : m_endpoint(endpoint)
    {
    }

    void AddRef() const { m_endpoint.AddRef(); }
    webrtc::RefCountReleaseStatus Release() const { return m_endpoint.Release(); }

private:
    void OnSetRemoteDescriptionComplete(webrtc::RTCError error) final
    {
        if (!error.ok()) {
            m_endpoint.setRemoteSessionDescriptionFailed(toExceptionCode(error.type()), error.message());
            return;
        }
        m_endpoint.setRemoteSessionDescriptionSucceeded();
    }

    Endpoint& m_endpoint;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(LIBWEBRTC)
