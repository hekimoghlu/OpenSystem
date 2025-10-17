/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 6, 2025.
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
#ifndef WebArchiveResource_h
#define WebArchiveResource_h

#if PLATFORM(COCOA)

#include "APIObject.h"
#include <wtf/Forward.h>
#include <wtf/RefPtr.h>

namespace API {
class Data;
class URL;
}

namespace WebCore {
class ArchiveResource;
}

namespace API {

class WebArchiveResource : public API::ObjectImpl<API::Object::Type::WebArchiveResource> {
public:
    virtual ~WebArchiveResource();

    static Ref<WebArchiveResource> create(API::Data*, const WTF::String& URL, const WTF::String& MIMEType, const WTF::String& textEncoding);
    static Ref<WebArchiveResource> create(RefPtr<WebCore::ArchiveResource>&&);

    Ref<API::Data> data();
    WTF::String url();
    WTF::String mimeType();
    WTF::String textEncoding();

    WebCore::ArchiveResource* coreArchiveResource();

private:
    WebArchiveResource(API::Data*, const WTF::String& URL, const WTF::String& MIMEType, const WTF::String& textEncoding);
    WebArchiveResource(RefPtr<WebCore::ArchiveResource>&&);

    RefPtr<WebCore::ArchiveResource> m_archiveResource;
};

} // namespace API

#endif // PLATFORM(COCOA)

#endif // WebArchiveResource_h
