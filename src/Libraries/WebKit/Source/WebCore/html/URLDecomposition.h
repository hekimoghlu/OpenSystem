/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 2, 2023.
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

#include <wtf/Forward.h>

namespace WebCore {

class URLDecomposition {
public:
    String origin() const;

    WEBCORE_EXPORT String protocol() const;
    void setProtocol(StringView);

    String username() const;
    void setUsername(StringView);

    String password() const;
    void setPassword(StringView);

    WEBCORE_EXPORT String host() const;
    void setHost(StringView);

    WEBCORE_EXPORT String hostname() const;
    void setHostname(StringView);

    WEBCORE_EXPORT String port() const;
    void setPort(StringView);

    WEBCORE_EXPORT String pathname() const;
    void setPathname(StringView);

    WEBCORE_EXPORT String search() const;
    void setSearch(const String&);

    WEBCORE_EXPORT String hash() const;
    void setHash(StringView);

protected:
    virtual ~URLDecomposition() = default;

private:
    virtual URL fullURL() const = 0;
    virtual void setFullURL(const URL&) = 0;
};

} // namespace WebCore
