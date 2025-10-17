/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 15, 2023.
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

#include "CodeOrigin.h"
#include "ExitingInlineKind.h"
#include <wtf/HashMap.h>

namespace JSC {

class CallLinkInfo;
class CallLinkStatus;
class CodeBlock;
class GetByStatus;
class InByStatus;
class PutByStatus;
class DeleteByStatus;
class StructureStubInfo;

struct ICStatus {
    StructureStubInfo* stubInfo { nullptr };
    CallLinkInfo* callLinkInfo { nullptr };
    CallLinkStatus* callStatus { nullptr };
    GetByStatus* getStatus { nullptr };
    InByStatus* inStatus { nullptr };
    PutByStatus* putStatus { nullptr };
    DeleteByStatus* deleteStatus { nullptr };
};

typedef UncheckedKeyHashMap<CodeOrigin, ICStatus, CodeOriginApproximateHash> ICStatusMap;

struct ICStatusContext {
    ICStatus get(CodeOrigin) const;
    bool isInlined(CodeOrigin) const;
    ExitingInlineKind inlineKind(CodeOrigin) const;
    
    InlineCallFrame* inlineCallFrame { nullptr };
    CodeBlock* optimizedCodeBlock { nullptr };
    ICStatusMap map;
};

typedef Vector<ICStatusContext*, 8> ICStatusContextStack;

} // namespace JSC

