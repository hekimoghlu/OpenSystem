/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 16, 2023.
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

#include "NodeFilter.h"
#include "NodeFilterCondition.h"
#include <wtf/Ref.h>

namespace WebCore {

class NativeNodeFilter final : public NodeFilter {
public:
    static Ref<NativeNodeFilter> create(ScriptExecutionContext* context, Ref<NodeFilterCondition>&& condition)
    {
        return adoptRef(*new NativeNodeFilter(context, WTFMove(condition)));
    }

    CallbackResult<unsigned short> acceptNode(Node&) override;
    CallbackResult<unsigned short> acceptNodeRethrowingException(Node&) override;

private:
    WEBCORE_EXPORT explicit NativeNodeFilter(ScriptExecutionContext*, Ref<NodeFilterCondition>&&);

    bool hasCallback() const final;

    Ref<NodeFilterCondition> m_condition;
};

} // namespace WebCore
