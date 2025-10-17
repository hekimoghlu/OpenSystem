/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 1, 2022.
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
#if ENABLE(ENCRYPTED_MEDIA)

#import <WebCore/MediaKeySystemClient.h>
#import <wtf/TZoneMalloc.h>

class WebMediaKeySystemClient final : public WebCore::MediaKeySystemClient {
    WTF_MAKE_TZONE_ALLOCATED(WebMediaKeySystemClient);
public:
    static WebMediaKeySystemClient& singleton();

private:
    friend NeverDestroyed<WebMediaKeySystemClient>;
    WebMediaKeySystemClient() = default;

    void pageDestroyed() override { }

    void requestMediaKeySystem(WebCore::MediaKeySystemRequest&) override;
    void cancelMediaKeySystemRequest(WebCore::MediaKeySystemRequest&) override { }
};

#endif // ENABLE(ENCRYPTED_MEDIA)
