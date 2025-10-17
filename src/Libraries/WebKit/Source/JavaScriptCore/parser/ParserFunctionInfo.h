/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 12, 2022.
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

namespace JSC {

template <class TreeBuilder>
struct ParserFunctionInfo {
    const Identifier* name = nullptr;
    typename TreeBuilder::FunctionBody body = 0;
    unsigned parameterCount = 0;
    unsigned functionLength = 0;
    unsigned startOffset = 0;
    unsigned endOffset = 0;
    int startLine = 0;
    int endLine = 0;
    unsigned parametersStartColumn = 0;
};

template <class TreeBuilder>
struct ParserClassInfo {
    const Identifier* className { nullptr };
    unsigned startOffset { 0 };
    unsigned endOffset { 0 };
    int startLine { 0 };
    unsigned startColumn { 0 };
};

}
