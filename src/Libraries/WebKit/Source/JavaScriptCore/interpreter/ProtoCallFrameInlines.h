/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 2, 2023.
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

#include "ProtoCallFrame.h"
#include "RegisterInlines.h"

namespace JSC {

inline void ProtoCallFrame::init(CodeBlock* codeBlock, JSGlobalObject* globalObject, JSObject* callee, JSValue thisValue, int argCountIncludingThis, EncodedJSValue* otherArgs)
{
    this->args = otherArgs;
    this->setCodeBlock(codeBlock);
    this->setCallee(callee);
    this->setGlobalObject(globalObject);
    this->setArgumentCountIncludingThis(argCountIncludingThis);
    size_t paddedArgsCount = argCountIncludingThis;
    if (codeBlock && static_cast<unsigned>(argCountIncludingThis) < codeBlock->numParameters())
        paddedArgsCount = codeBlock->numParameters();
    paddedArgsCount = roundArgumentCountToAlignFrame(paddedArgsCount);
    this->setPaddedArgCount(paddedArgsCount);
    this->clearCurrentVPC();
    this->setThisValue(thisValue);
}

inline JSObject* ProtoCallFrame::callee() const
{
    return calleeValue.Register::object();
}

inline void ProtoCallFrame::setCallee(JSObject* callee)
{
    calleeValue = callee;
}

inline CodeBlock* ProtoCallFrame::codeBlock() const
{
    return codeBlockValue.Register::codeBlock();
}

inline void ProtoCallFrame::setCodeBlock(CodeBlock* codeBlock)
{
    codeBlockValue = codeBlock;
}

} // namespace JSC
