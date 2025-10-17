/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 29, 2023.
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

#if ENABLE(JIT)

#include "CallMode.h"
#include "CodeSpecializationKind.h"
#include "JSCPtrTag.h"

namespace JSC {

class CallLinkInfo;
enum class CallMode;
template<PtrTag> class MacroAssemblerCodeRef;
class VM;

MacroAssemblerCodeRef<JITThunkPtrTag> handleExceptionGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> handleExceptionWithCallFrameRollbackGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> popThunkStackPreservesAndHandleExceptionGenerator(VM&);

MacroAssemblerCodeRef<JITThunkPtrTag> throwExceptionFromCallGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> throwExceptionFromCallSlowPathGenerator(VM&);

MacroAssemblerCodeRef<JITThunkPtrTag> checkExceptionGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> returnFromBaselineGenerator(VM&);

MacroAssemblerCodeRef<JITThunkPtrTag> polymorphicThunk(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> polymorphicThunkForClosure(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> polymorphicTopTierThunk(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> polymorphicTopTierThunkForClosure(VM&);

MacroAssemblerCodeRef<JITThunkPtrTag> virtualThunkForRegularCall(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> virtualThunkForTailCall(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> virtualThunkForConstruct(VM&);

MacroAssemblerCodeRef<JITThunkPtrTag> nativeCallGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> nativeCallWithDebuggerHookGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> nativeConstructGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> nativeConstructWithDebuggerHookGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> nativeTailCallGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> nativeTailCallWithoutSavedTagsGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> internalFunctionCallGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> internalFunctionConstructGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> arityFixupGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> unreachableGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> stringGetByValGenerator(VM&);

MacroAssemblerCodeRef<JITThunkPtrTag> charCodeAtThunkGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> charAtThunkGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> stringPrototypeCodePointAtThunkGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> clz32ThunkGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> fromCharCodeThunkGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> globalIsNaNThunkGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> numberIsNaNThunkGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> absThunkGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> ceilThunkGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> expThunkGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> floorThunkGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> logThunkGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> roundThunkGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> sqrtThunkGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> imulThunkGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> randomThunkGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> truncThunkGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> numberConstructorCallThunkGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> stringConstructorCallThunkGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> boundFunctionCallGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> remoteFunctionCallGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> toIntegerOrInfinityThunkGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> toLengthThunkGenerator(VM&);
#if CPU(ARM64)
MacroAssemblerCodeRef<JITThunkPtrTag> maxThunkGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> minThunkGenerator(VM&);
#endif

#if USE(JSVALUE64)
MacroAssemblerCodeRef<JITThunkPtrTag> objectIsThunkGenerator(VM&);
#endif

} // namespace JSC
#endif // ENABLE(JIT)
