/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 30, 2022.
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

#if ENABLE(FTL_JIT)

#include "DFGOperations.h"
#include "FTLExitTimeObjectMaterialization.h"

namespace JSC {

struct UnlinkedStringJumpTable;

namespace FTL {

class LazySlowPath;

JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationMaterializeObjectInOSR, JSCell*, (JSGlobalObject*, ExitTimeObjectMaterialization*, EncodedJSValue*));

JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationPopulateObjectInOSR, void, (JSGlobalObject*, ExitTimeObjectMaterialization*, EncodedJSValue*, EncodedJSValue*));

JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationCompileFTLLazySlowPath, void*, (CallFrame*, unsigned));

JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationSwitchStringAndGetIndex, unsigned, (JSGlobalObject*, const UnlinkedStringJumpTable*, JSString*));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationTypeOfObjectAsTypeofType, int32_t, (JSGlobalObject*, JSCell*));

JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationReportBoundsCheckEliminationErrorAndCrash, void, (intptr_t codeBlockAsIntPtr, int32_t, int32_t, int32_t, int32_t, int32_t));

} } // namespace JSC::DFG

#endif // ENABLE(FTL_JIT)
