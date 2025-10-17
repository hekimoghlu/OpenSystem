/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 21, 2023.
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

#if USE(QUICK_LOOK)

#include <WebCore/LegacyPreviewLoaderClient.h>
#include <WebCore/PageIdentifier.h>
#include <WebCore/SharedBuffer.h>
#include <wtf/text/WTFString.h>

namespace WebKit {

class WebFrame;

class WebPreviewLoaderClient final : public WebCore::LegacyPreviewLoaderClient {
public:
    static Ref<WebPreviewLoaderClient> create(const String& fileName, const String& uti, WebCore::PageIdentifier pageID)
    {
        return adoptRef(*new WebPreviewLoaderClient(fileName, uti, pageID));
    }
    ~WebPreviewLoaderClient();

private:
    WebPreviewLoaderClient(const String& fileName, const String& uti, WebCore::PageIdentifier);
    void didReceiveData(const WebCore::SharedBuffer&) override;
    void didFinishLoading() override;
    void didFail() override;
    bool supportsPasswordEntry() const override { return true; }
    void didRequestPassword(Function<void(const String&)>&&) override;

    const String m_fileName;
    const String m_uti;
    const WebCore::PageIdentifier m_pageID;
    WebCore::SharedBufferBuilder m_buffer;
};

} // namespace WebKit

#endif // USE(QUICK_LOOK)
