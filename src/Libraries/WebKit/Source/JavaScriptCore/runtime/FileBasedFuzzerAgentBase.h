/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 12, 2025.
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

#include "ExpressionInfo.h"
#include "FuzzerAgent.h"
#include "LineColumn.h"
#include "Opcode.h"
#include <wtf/Lock.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {

class VM;

struct PredictionTarget {
    ExpressionInfo::Entry info;
    OpcodeID opcodeId;
    String sourceFilename;
    String lookupKey;
};

class FileBasedFuzzerAgentBase : public FuzzerAgent {
    WTF_MAKE_TZONE_ALLOCATED(FileBasedFuzzerAgentBase);

public:
    FileBasedFuzzerAgentBase(VM&);

protected:
    Lock m_lock;
    virtual SpeculatedType getPredictionInternal(CodeBlock*, PredictionTarget&, SpeculatedType original) = 0;

public:
    SpeculatedType getPrediction(CodeBlock*, const CodeOrigin&, SpeculatedType original) final;

protected:
    static String createLookupKey(const String& sourceFilename, OpcodeID, int startLocation, int endLocation);
    static OpcodeID opcodeAliasForLookupKey(const OpcodeID&);
};

} // namespace JSC
