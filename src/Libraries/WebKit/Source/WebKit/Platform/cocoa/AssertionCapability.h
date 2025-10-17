/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 9, 2023.
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

#if ENABLE(EXTENSION_CAPABILITIES)

#include "ExtensionCapability.h"
#include <wtf/BlockPtr.h>
#include <wtf/Forward.h>
#include <wtf/text/WTFString.h>

namespace WebKit {

class AssertionCapability final : public ExtensionCapability {
public:
    AssertionCapability(String environmentIdentifier, String domain, String name, Function<void()>&& willInvalidateFunction = nullptr, Function<void()>&& didInvalidateFunction = nullptr);

    const String& domain() const { return m_domain; }
    const String& name() const { return m_name; }

    // ExtensionCapability
    String environmentIdentifier() const final { return m_environmentIdentifier; }

    BlockPtr<void()> didInvalidateBlock() const { return m_didInvalidateBlock; }

private:
    String m_environmentIdentifier;
    String m_domain;
    String m_name;
    BlockPtr<void()> m_willInvalidateBlock;
    BlockPtr<void()> m_didInvalidateBlock;
};

} // namespace WebKit

#endif // ENABLE(EXTENSION_CAPABILITIES)
