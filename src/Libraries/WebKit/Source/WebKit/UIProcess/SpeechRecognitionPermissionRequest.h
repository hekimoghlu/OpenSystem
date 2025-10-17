/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 7, 2021.
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

#include "APIObject.h"
#include <WebCore/ClientOrigin.h>
#include <WebCore/FrameIdentifier.h>
#include <WebCore/SpeechRecognitionError.h>
#include <WebCore/SpeechRecognitionRequest.h>
#include <wtf/CompletionHandler.h>

namespace WebKit {

using SpeechRecognitionPermissionRequestCallback = CompletionHandler<void(std::optional<WebCore::SpeechRecognitionError>&&)>;

class SpeechRecognitionPermissionRequest : public RefCounted<SpeechRecognitionPermissionRequest> {
public:
    static Ref<SpeechRecognitionPermissionRequest> create(WebCore::SpeechRecognitionRequest& request, SpeechRecognitionPermissionRequestCallback&& completionHandler)
    {
        return adoptRef(*new SpeechRecognitionPermissionRequest(request, WTFMove(completionHandler)));
    }

    ~SpeechRecognitionPermissionRequest()
    {
        if (m_completionHandler)
            m_completionHandler(WebCore::SpeechRecognitionError { WebCore::SpeechRecognitionErrorType::NotAllowed, "Request is cancelled"_s });
    }

    void complete(std::optional<WebCore::SpeechRecognitionError>&& error)
    {
        auto completionHandler = std::exchange(m_completionHandler, { });
        completionHandler(WTFMove(error));
    }

    WebCore::SpeechRecognitionRequest* request() { return m_request.get(); }

private:
    SpeechRecognitionPermissionRequest(WebCore::SpeechRecognitionRequest& request, SpeechRecognitionPermissionRequestCallback&& completionHandler)
        : m_request(request)
        , m_completionHandler(WTFMove(completionHandler))
    { }


    WeakPtr<WebCore::SpeechRecognitionRequest> m_request;
    SpeechRecognitionPermissionRequestCallback m_completionHandler;
};

class SpeechRecognitionPermissionCallback : public API::ObjectImpl<API::Object::Type::SpeechRecognitionPermissionCallback> {
public:
    static Ref<SpeechRecognitionPermissionCallback> create(CompletionHandler<void(bool)>&& completionHandler)
    {
        return adoptRef(*new SpeechRecognitionPermissionCallback(WTFMove(completionHandler)));
    }

    void complete(bool granted) { m_completionHandler(granted); }

private:
    SpeechRecognitionPermissionCallback(CompletionHandler<void(bool)>&& completionHandler)
        : m_completionHandler(WTFMove(completionHandler))
    { }

    CompletionHandler<void(bool)> m_completionHandler;
};

} // namespace WebKit
