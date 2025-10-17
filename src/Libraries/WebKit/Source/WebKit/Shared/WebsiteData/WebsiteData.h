/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 28, 2023.
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

#include <WebCore/RegistrableDomain.h>
#include <WebCore/SecurityOriginData.h>
#include <wtf/HashSet.h>
#include <wtf/OptionSet.h>
#include <wtf/Vector.h>

namespace IPC {
class Decoder;
class Encoder;
}

namespace WebKit {

enum class WebsiteDataType : uint32_t;

enum class WebsiteDataProcessType { Network, UI, Web };

struct WebsiteData {
    struct Entry {
        Entry(WebCore::SecurityOriginData, WebsiteDataType, uint64_t);
        Entry(WebCore::SecurityOriginData&&, OptionSet<WebsiteDataType>&&, uint64_t);

        OptionSet<WebsiteDataType> typeAsOptionSet() const { return { type }; }
        
        Entry isolatedCopy() const &;
        Entry isolatedCopy() &&;

        WebCore::SecurityOriginData origin;
        WebsiteDataType type;
        uint64_t size;
    };

    WebsiteData isolatedCopy() const &;
    WebsiteData isolatedCopy() &&;

    Vector<Entry> entries;
    HashSet<String> hostNamesWithCookies;

    HashSet<String> hostNamesWithHSTSCache;
    HashSet<WebCore::RegistrableDomain> registrableDomainsWithResourceLoadStatistics;
    static WebsiteDataProcessType ownerProcess(WebsiteDataType);
    static OptionSet<WebsiteDataType> filter(OptionSet<WebsiteDataType>, WebsiteDataProcessType);
};

}
