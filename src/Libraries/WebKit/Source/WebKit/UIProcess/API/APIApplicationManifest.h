/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 2, 2023.
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

#if ENABLE(APPLICATION_MANIFEST)

#include "APIObject.h"
#include <WebCore/ApplicationManifest.h>

namespace API {

class ApplicationManifest final : public ObjectImpl<Object::Type::ApplicationManifest> {
public:
    static Ref<ApplicationManifest> create(const WebCore::ApplicationManifest& applicationManifest)
    {
        return adoptRef(*new ApplicationManifest(applicationManifest));
    }

    explicit ApplicationManifest(const WebCore::ApplicationManifest& applicationManifest)
        : m_applicationManifest(applicationManifest)
    {
    }

    const WebCore::ApplicationManifest& applicationManifest() const { return m_applicationManifest; }

private:
    WebCore::ApplicationManifest m_applicationManifest;
};

} // namespace API

#endif // ENABLE(APPLICATION_MANIFEST)
