/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 15, 2024.
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

#include "CompiledContentExtension.h"
#include <system_error>
#include <wtf/Forward.h>
#include <wtf/Ref.h>

namespace WebCore::ContentExtensions {

class ContentExtensionCompilationClient {
public:
    virtual ~ContentExtensionCompilationClient() = default;
    
    // Functions should be called in this order. All except writeActions and finalize can be called multiple times, though.
    virtual void writeSource(String&&) = 0;
    virtual void writeActions(Vector<SerializedActionByte>&&) = 0;
    virtual void writeURLFiltersBytecode(Vector<DFABytecode>&&) = 0;
    virtual void writeTopURLFiltersBytecode(Vector<DFABytecode>&&) = 0;
    virtual void writeFrameURLFiltersBytecode(Vector<DFABytecode>&&) = 0;
    virtual void finalize() = 0;
};

WEBCORE_EXPORT std::error_code compileRuleList(ContentExtensionCompilationClient&, String&& ruleJSON, Vector<ContentExtensionRule>&&);

} // namespace WebCore::ContentExtensions

#endif // ENABLE(CONTENT_EXTENSIONS)
