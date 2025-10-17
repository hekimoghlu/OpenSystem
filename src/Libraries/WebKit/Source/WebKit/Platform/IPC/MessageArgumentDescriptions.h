/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 29, 2025.
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

#if ENABLE(IPC_TESTING_API) || !LOG_DISABLED

#include "MessageNames.h"
#include <wtf/Vector.h>

namespace JSC {

class JSGlobalObject;
class JSValue;

}

namespace IPC {

class Decoder;

#if ENABLE(IPC_TESTING_API)

std::optional<JSC::JSValue> jsValueForArguments(JSC::JSGlobalObject*, MessageName, Decoder&);
std::optional<JSC::JSValue> jsValueForReplyArguments(JSC::JSGlobalObject*, MessageName, Decoder&);

Vector<ASCIILiteral> serializedIdentifiers();

#endif // ENABLE(IPC_TESTING_API)

struct ArgumentDescription {
    ASCIILiteral name;
    ASCIILiteral type;
    ASCIILiteral enumName;
    bool isOptional;
};

std::optional<Vector<ArgumentDescription>> messageArgumentDescriptions(MessageName);
std::optional<Vector<ArgumentDescription>> messageReplyArgumentDescriptions(MessageName);

}

#endif // ENABLE(IPC_TESTING_API) || !LOG_DISABLED
