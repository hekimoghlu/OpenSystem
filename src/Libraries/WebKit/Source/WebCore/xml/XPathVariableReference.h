/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 14, 2023.
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

#include "XPathExpressionNode.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {
namespace XPath {

// Variable references are not used with XPathEvaluator.
class VariableReference : public Expression {
    WTF_MAKE_TZONE_ALLOCATED(VariableReference);
public:
    explicit VariableReference(const String& name);
private:
    Value evaluate() const override;
    Value::Type resultType() const override { ASSERT_NOT_REACHED(); return Value::Type::Number; }
    String m_name;
};

} // namespace XPath
} // namespace WebCore
