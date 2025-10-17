/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 28, 2022.
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

#if ENABLE(MEDIA_STREAM)

#include "MediaProducer.h"
#include <wtf/CompletionHandler.h>
#include <wtf/ObjectIdentifier.h>

namespace WebCore {

struct CaptureDeviceWithCapabilities;
class Document;
class Exception;
class Page;
class UserMediaRequest;

struct MediaDeviceHashSalts;

class UserMediaClient {
public:
    virtual void pageDestroyed() = 0;

    virtual void requestUserMediaAccess(UserMediaRequest&) = 0;
    virtual void cancelUserMediaAccessRequest(UserMediaRequest&) = 0;

    using EnumerateDevicesCallback = CompletionHandler<void(Vector<CaptureDeviceWithCapabilities>&&, MediaDeviceHashSalts&&)>;
    virtual void enumerateMediaDevices(Document&, EnumerateDevicesCallback&&) = 0;

    enum DeviceChangeObserverTokenType { };
    using DeviceChangeObserverToken = ObjectIdentifier<DeviceChangeObserverTokenType>;
    virtual DeviceChangeObserverToken addDeviceChangeObserver(Function<void()>&&) = 0;
    virtual void removeDeviceChangeObserver(DeviceChangeObserverToken) = 0;

    virtual void updateCaptureState(const Document&, bool isActive, MediaProducerMediaCaptureKind, CompletionHandler<void(std::optional<Exception>&&)>&&) = 0;
    virtual void setShouldListenToVoiceActivity(bool) = 0;

protected:
    virtual ~UserMediaClient() = default;
};

WEBCORE_EXPORT void provideUserMediaTo(Page*, UserMediaClient*);

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
