/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 13, 2023.
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

class Function : public Expression {
    WTF_MAKE_TZONE_ALLOCATED(Function);
public:
    static std::unique_ptr<Function> create(const String& name);
    static std::unique_ptr<Function> create(const String& name, Vector<std::unique_ptr<Expression>> arguments);

protected:
    unsigned argumentCount() const { return subexpressionCount(); }
    const Expression& argument(unsigned i) const { return subexpression(i); }

private:
    static std::unique_ptr<Function> create(const String& name, unsigned numArguments);
    void setArguments(const String& name, Vector<std::unique_ptr<Expression>>);
};

} // namespace XPath
} // namespace WebCore
