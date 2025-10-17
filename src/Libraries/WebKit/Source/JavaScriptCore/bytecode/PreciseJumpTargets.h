/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 5, 2023.
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

#include "CodeBlock.h"

namespace JSC {

class UnlinkedCodeBlock;

// Return a sorted list of bytecode index that are the destination of a jump.
void computePreciseJumpTargets(CodeBlock*, Vector<JSInstructionStream::Offset, 32>& out);
void computePreciseJumpTargets(CodeBlock*, const JSInstructionStream& instructions, Vector<JSInstructionStream::Offset, 32>& out);
void computePreciseJumpTargets(UnlinkedCodeBlockGenerator*, const JSInstructionStream&, Vector<JSInstructionStream::Offset, 32>& out);

void recomputePreciseJumpTargets(UnlinkedCodeBlockGenerator*, const JSInstructionStream&, Vector<JSInstructionStream::Offset>& out);

void findJumpTargetsForInstruction(CodeBlock*, const JSInstructionStream::Ref&, Vector<JSInstructionStream::Offset, 1>& out);
void findJumpTargetsForInstruction(UnlinkedCodeBlockGenerator*, const JSInstructionStream::Ref&, Vector<JSInstructionStream::Offset, 1>& out);

} // namespace JSC
