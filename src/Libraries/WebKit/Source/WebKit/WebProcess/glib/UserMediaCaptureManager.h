/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 5, 2023.
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

#if USE(GLIB) && ENABLE(MEDIA_STREAM)

#include "MessageReceiver.h"
#include "WebProcessSupplement.h"
#include <wtf/CheckedRef.h>
#include <wtf/CompletionHandler.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class CaptureDevice;
struct CaptureDeviceWithCapabilities;
struct MediaDeviceHashSalts;
struct MediaStreamRequest;

enum class MediaConstraintType : uint8_t;
}

namespace WebKit {

class WebProcess;

class UserMediaCaptureManager : public WebProcessSupplement, public IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(UserMediaCaptureManager);
    WTF_MAKE_NONCOPYABLE(UserMediaCaptureManager);
public:
    explicit UserMediaCaptureManager(WebProcess&);
    ~UserMediaCaptureManager();

    void ref() const final;
    void deref() const final;

    static ASCIILiteral supplementName() { return "UserMediaCaptureManager"_s; }

private:
    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    // Messages::UserMediaCaptureManager
    using ValidateUserMediaRequestConstraintsCallback = CompletionHandler<void(std::optional<WebCore::MediaConstraintType> invalidConstraint, Vector<WebCore::CaptureDevice>& audioDevices, Vector<WebCore::CaptureDevice>& videoDevices)>;
    void validateUserMediaRequestConstraints(WebCore::MediaStreamRequest, WebCore::MediaDeviceHashSalts&&, ValidateUserMediaRequestConstraintsCallback&&);
    ValidateUserMediaRequestConstraintsCallback m_validateUserMediaRequestConstraintsCallback;

    using GetMediaStreamDevicesCallback = CompletionHandler<void(Vector<WebCore::CaptureDeviceWithCapabilities>&&)>;
    void getMediaStreamDevices(bool revealIdsAndLabels, GetMediaStreamDevicesCallback&&);

    CheckedRef<WebProcess> m_process;
};

} // namespace WebKit

#endif
