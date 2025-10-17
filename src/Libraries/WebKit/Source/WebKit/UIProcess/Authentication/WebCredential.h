/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 11, 2022.
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
#ifndef WebCredential_h
#define WebCredential_h

#include "APIObject.h"
#include "APIString.h"
#include <WebCore/Credential.h>

namespace WebKit {

class WebCredential : public API::ObjectImpl<API::Object::Type::Credential> {
public:
    ~WebCredential();

    static Ref<WebCredential> create(const WebCore::Credential& credential)
    {
        return adoptRef(*new WebCredential(credential));
    }

    const WebCore::Credential& credential();

private:
    explicit WebCredential(const WebCore::Credential&);

    WebCore::Credential m_coreCredential;
};

} // namespace WebKit

#endif // WebCredential_h
