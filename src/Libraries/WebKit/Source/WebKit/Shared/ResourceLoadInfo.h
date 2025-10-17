/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 29, 2022.
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

#include "ArgumentCoders.h"
#include "NetworkResourceLoadIdentifier.h"
#include <WebCore/FrameIdentifier.h>
#include <wtf/URL.h>
#include <wtf/UUID.h>
#include <wtf/WallTime.h>

namespace WebKit {

struct ResourceLoadInfo {
    enum class Type : uint8_t {
        ApplicationManifest,
        Beacon,
        CSPReport,
        Document,
        Fetch,
        Font,
        Image,
        Media,
        Object,
        Other,
        Ping,
        Script,
        Stylesheet,
        XMLHTTPRequest,
        XSLT
    };

    NetworkResourceLoadIdentifier resourceLoadID;
    std::optional<WebCore::FrameIdentifier> frameID;
    std::optional<WebCore::FrameIdentifier> parentFrameID;
    Markable<WTF::UUID> documentID;
    URL originalURL;
    String originalHTTPMethod;
    WallTime eventTimestamp;
    bool loadedFromCache { false };
    Type type { Type::Other };
};

} // namespace WebKit
