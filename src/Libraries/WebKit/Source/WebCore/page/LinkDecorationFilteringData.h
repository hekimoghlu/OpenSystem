/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

#include "RegistrableDomain.h"

namespace WebCore {

struct LinkDecorationFilteringData {
    RegistrableDomain domain;
    String path;
    String linkDecoration;

    LinkDecorationFilteringData(RegistrableDomain&& domain, String&& path, String&& linkDecoration)
        : domain(WTFMove(domain))
        , path(WTFMove(path))
        , linkDecoration(WTFMove(linkDecoration))
    {
    }

    LinkDecorationFilteringData(String&& domain, String&& path, String&& linkDecoration)
        : LinkDecorationFilteringData(RegistrableDomain { URL { WTFMove(domain) } }, WTFMove(path), WTFMove(linkDecoration))
    {
    }

    LinkDecorationFilteringData(const LinkDecorationFilteringData&) = default;
    LinkDecorationFilteringData& operator=(const LinkDecorationFilteringData&) = default;

    LinkDecorationFilteringData(LinkDecorationFilteringData&& data)
        : domain(WTFMove(data.domain))
        , path(WTFMove(data.path))
        , linkDecoration(WTFMove(data.linkDecoration))
    {
    }

    LinkDecorationFilteringData& operator=(LinkDecorationFilteringData&& data)
    {
        domain = WTFMove(data.domain);
        path = WTFMove(data.path);
        linkDecoration = WTFMove(data.linkDecoration);
        return *this;
    }
};

}
