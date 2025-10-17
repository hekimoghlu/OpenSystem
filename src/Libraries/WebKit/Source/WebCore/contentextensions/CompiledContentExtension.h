/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 28, 2022.
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

#if ENABLE(CONTENT_EXTENSIONS)

#include "ContentExtensionRule.h"
#include "DFABytecode.h"
#include <span>
#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore::ContentExtensions {

class WEBCORE_EXPORT CompiledContentExtension : public ThreadSafeRefCounted<CompiledContentExtension> {
public:
    virtual ~CompiledContentExtension();

    virtual std::span<const uint8_t> urlFiltersBytecode() const = 0;
    virtual std::span<const uint8_t> topURLFiltersBytecode() const = 0;
    virtual std::span<const uint8_t> frameURLFiltersBytecode() const = 0;
    virtual std::span<const uint8_t> serializedActions() const = 0;
};

} // namespace WebCore::ContentExtensions

#endif // ENABLE(CONTENT_EXTENSIONS)
