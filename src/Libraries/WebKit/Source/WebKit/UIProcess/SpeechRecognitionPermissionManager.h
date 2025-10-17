/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 4, 2025.
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

#include "SpeechRecognitionPermissionRequest.h"
#include <wtf/Deque.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebKit {

class WebPageProxy;

class SpeechRecognitionPermissionManager : public RefCountedAndCanMakeWeakPtr<SpeechRecognitionPermissionManager> {
    WTF_MAKE_TZONE_ALLOCATED(SpeechRecognitionPermissionManager);
public:
    enum class CheckResult { Denied, Granted, Unknown };
    static Ref<SpeechRecognitionPermissionManager> create(WebPageProxy&);
    ~SpeechRecognitionPermissionManager();

    void request(WebCore::SpeechRecognitionRequest&, SpeechRecognitionPermissionRequestCallback&&);

    void decideByDefaultAction(const WebCore::SecurityOriginData&, CompletionHandler<void(bool)>&&);
    WebPageProxy* page();

private:
    explicit SpeechRecognitionPermissionManager(WebPageProxy&);
    RefPtr<WebPageProxy> protectedPage() const;

    void startNextRequest();
    void startProcessingRequest();
    void continueProcessingRequest();
    void completeCurrentRequest(std::optional<WebCore::SpeechRecognitionError>&& = std::nullopt);
    void requestMicrophoneAccess();
    void requestSpeechRecognitionServiceAccess();
    void requestUserPermission(WebCore::SpeechRecognitionRequest& request);

    WeakPtr<WebPageProxy> m_page;
    Deque<Ref<SpeechRecognitionPermissionRequest>> m_requests;
    CheckResult m_microphoneCheck { CheckResult::Unknown };
    CheckResult m_speechRecognitionServiceCheck { CheckResult::Unknown };
    CheckResult m_userPermissionCheck { CheckResult::Unknown };
};

} // namespace WebKit
