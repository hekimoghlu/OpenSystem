/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 6, 2024.
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
#ifndef APIWebsiteDataRecord_h
#define APIWebsiteDataRecord_h

#include "APIObject.h"
#include "WebsiteDataRecord.h"

namespace API {

class WebsiteDataRecord final : public ObjectImpl<Object::Type::WebsiteDataRecord> {
public:
    static Ref<WebsiteDataRecord> create(WebKit::WebsiteDataRecord&&);
    virtual ~WebsiteDataRecord();

    const WebKit::WebsiteDataRecord& websiteDataRecord() const { return m_websiteDataRecord; }

private:
    explicit WebsiteDataRecord(WebKit::WebsiteDataRecord&&);

    const WebKit::WebsiteDataRecord m_websiteDataRecord;
};

}

#endif // APIWebsiteDataRecord_h
