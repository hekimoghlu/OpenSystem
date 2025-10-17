/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 7, 2022.
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

#if USE(CURL) || USE(SOUP)

#include "NetworkDataTask.h"
#include <WebCore/DataURLDecoder.h>
#include <WebCore/ResourceResponse.h>
#include <wtf/FileSystem.h>
#include <wtf/Forward.h>

namespace WebKit {

class Download;

class NetworkDataTaskDataURL : public NetworkDataTask {
public:
    static Ref<NetworkDataTask> create(NetworkSession&, NetworkDataTaskClient&, const NetworkLoadParameters&);

    ~NetworkDataTaskDataURL() override;

private:
    NetworkDataTaskDataURL(NetworkSession&, NetworkDataTaskClient&, const NetworkLoadParameters&);

    void cancel() override;
    void resume() override;
    void invalidateAndCancel() override;
    State state() const override;

    void setPendingDownloadLocation(const String& filename, SandboxExtension::Handle&&, bool /*allowOverwrite*/) override;
    String suggestedFilename() const override;

    void didDecodeDataURL(std::optional<WebCore::DataURLDecoder::Result>&&);
    void downloadDecodedData(Vector<uint8_t>&&);

    State m_state { State::Suspended };
    bool m_allowOverwriteDownload { false };
    WebCore::ResourceResponse m_response;
};

} // namespace WebKit

#endif // USE(CURL) || USE(SOUP)
