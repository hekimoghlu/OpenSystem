/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 4, 2024.
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

#if USE(QUICK_LOOK)

#include "LegacyPreviewLoaderClient.h"

namespace WebCore {

class MockPreviewLoaderClient final : public LegacyPreviewLoaderClient {
public:
    static MockPreviewLoaderClient& singleton();

    void setPassword(const String& password) { m_password = password; }

    bool supportsPasswordEntry() const override { return true; }
    void didRequestPassword(Function<void(const String&)>&&) override;

private:
    friend class NeverDestroyed<MockPreviewLoaderClient>;
    MockPreviewLoaderClient() = default;

    String m_password;
};

} // namespace WebCore

#endif // USE(QUICK_LOOK)
