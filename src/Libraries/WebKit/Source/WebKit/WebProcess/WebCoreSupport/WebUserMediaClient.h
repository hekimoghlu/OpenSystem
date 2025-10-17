/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 8, 2023.
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
#ifndef WebUserMediaClient_h
#define WebUserMediaClient_h

#if ENABLE(MEDIA_STREAM)

#include <WebCore/UserMediaClient.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>

namespace WebKit {

class WebPage;

class WebUserMediaClient : public WebCore::UserMediaClient {
    WTF_MAKE_TZONE_ALLOCATED(WebUserMediaClient);
public:
    WebUserMediaClient(WebPage&);
    ~WebUserMediaClient() { }

private:
    Ref<WebPage> protectedPage() const;

    void pageDestroyed() override;

    void requestUserMediaAccess(WebCore::UserMediaRequest&) override;
    void cancelUserMediaAccessRequest(WebCore::UserMediaRequest&) override;

    void enumerateMediaDevices(WebCore::Document&, WebCore::UserMediaClient::EnumerateDevicesCallback&&) final;

    DeviceChangeObserverToken addDeviceChangeObserver(WTF::Function<void()>&&) final;
    void removeDeviceChangeObserver(DeviceChangeObserverToken) final;
    void updateCaptureState(const WebCore::Document&, bool isActive, WebCore::MediaProducerMediaCaptureKind, CompletionHandler<void(std::optional<WebCore::Exception>&&)>&&) final;
    void setShouldListenToVoiceActivity(bool) final;

    void initializeFactories();

    WeakRef<WebPage> m_page;
};

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)

#endif // WebUserMediaClient_h
