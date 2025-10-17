/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 6, 2022.
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
#import <WebCore/GeolocationClient.h>
#import <wtf/TZoneMalloc.h>

namespace WebCore {
class Geolocation;
class GeolocationPositionData;
}

@class WebView;

class WebGeolocationClient : public WebCore::GeolocationClient {
    WTF_MAKE_TZONE_ALLOCATED(WebGeolocationClient);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(WebGeolocationClient);
public:
    WebGeolocationClient(WebView *);
    WebView *webView() { return m_webView; }

    void geolocationDestroyed() override;
    void startUpdating(const String& authorizationToken, bool enableHighAccuracy) override;
    void stopUpdating() override;
#if PLATFORM(IOS_FAMILY)
    // FIXME: unify this with Mac on OpenSource.
    void setEnableHighAccuracy(bool) override;
#else
    void setEnableHighAccuracy(bool) override { }
#endif

    std::optional<WebCore::GeolocationPositionData> lastPosition() override;

    void requestPermission(WebCore::Geolocation&) override;
    void cancelPermissionRequest(WebCore::Geolocation&) override { };

private:
    WebView *m_webView;
};
