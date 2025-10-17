/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 9, 2023.
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

#include <wtf/NeverDestroyed.h>
#include <wtf/ObjectIdentifier.h>

namespace WebKit {

enum class ContentWorldIdentifierType { };
using ContentWorldIdentifier = ObjectIdentifier<ContentWorldIdentifierType>;

inline ContentWorldIdentifier pageContentWorldIdentifier()
{
    static NeverDestroyed<ContentWorldIdentifier> identifier(ObjectIdentifier<ContentWorldIdentifierType>(1));
    return identifier;
}

enum class ContentWorldOption : uint8_t {
    AllowAccessToClosedShadowRoots = 1 << 0,
    AllowAutofill = 1 << 1,
    AllowElementUserInfo = 1 << 2,
    DisableLegacyBuiltinOverrides = 1 << 3,
};

} // namespace WebKit
