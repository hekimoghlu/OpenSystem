/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 20, 2023.
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

#include "WebProcessSupplement.h"
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WallTime.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class SecurityOriginData;
}

namespace WebKit {

class WebProcess;

class WebMediaKeyStorageManager : public WebProcessSupplement {
    WTF_MAKE_TZONE_ALLOCATED(WebMediaKeyStorageManager);
    WTF_MAKE_NONCOPYABLE(WebMediaKeyStorageManager);
public:
    explicit WebMediaKeyStorageManager(WebProcess&) { }
    virtual ~WebMediaKeyStorageManager() { }

    static ASCIILiteral supplementName();

    const String& mediaKeyStorageDirectory() const { return m_mediaKeyStorageDirectory; }
    String mediaKeyStorageDirectoryForOrigin(const WebCore::SecurityOriginData&);

    Vector<WebCore::SecurityOriginData> getMediaKeyOrigins();
    void deleteMediaKeyEntriesForOrigin(const WebCore::SecurityOriginData&);
    void deleteMediaKeyEntriesModifiedBetweenDates(WallTime startDate, WallTime endDate);
    void deleteAllMediaKeyEntries();

private:
    void setWebsiteDataStore(const WebProcessDataStoreParameters&) override;

    String m_mediaKeyStorageDirectory;
};

}
