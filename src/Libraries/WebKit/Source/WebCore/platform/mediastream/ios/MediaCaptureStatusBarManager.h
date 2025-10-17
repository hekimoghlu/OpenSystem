/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 24, 2023.
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

#if ENABLE(MEDIA_STREAM) && PLATFORM(IOS_FAMILY)

#include <wtf/CompletionHandler.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

OBJC_CLASS WebCoreMediaCaptureStatusBarHandler;

namespace WebCore {
class MediaCaptureStatusBarManager;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::MediaCaptureStatusBarManager> : std::true_type { };
}

namespace WebCore {

class MediaCaptureStatusBarManager : public CanMakeWeakPtr<MediaCaptureStatusBarManager> {
    WTF_MAKE_TZONE_ALLOCATED(MediaCaptureStatusBarManager);
public:
    static bool hasSupport();

    using TapCallback = Function<void(CompletionHandler<void()>&&)>;
    using ErrorCallback = Function<void()>;
    MediaCaptureStatusBarManager(TapCallback&& callback, ErrorCallback&& errorCallback)
        : m_tapCallback(WTFMove(callback))
        , m_errorCallback(WTFMove(errorCallback))
    {
    }
    ~MediaCaptureStatusBarManager();

    void start();
    void stop();

    void didError() { m_errorCallback(); }
    void didTap(CompletionHandler<void()>&& completionHandler) { m_tapCallback(WTFMove(completionHandler)); }

private:
    RetainPtr<WebCoreMediaCaptureStatusBarHandler> m_handler;
    TapCallback m_tapCallback;
    ErrorCallback m_errorCallback;
};

}

#endif // ENABLE(MEDIA_STREAM) && PLATFORM(IOS_FAMILY)
