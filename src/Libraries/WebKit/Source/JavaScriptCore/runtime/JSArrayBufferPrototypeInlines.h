/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 15, 2023.
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

#include "JSArrayBuffer.h"
#include "JSArrayBufferPrototype.h"
#include "JSGlobalObject.h"

namespace JSC  {

namespace JSArrayBufferPrototypeInternal {
static constexpr bool verbose = false;
};

ALWAYS_INLINE bool speciesWatchpointIsValid(JSObject* thisObject, ArrayBufferSharingMode mode)
{
    JSGlobalObject* globalObject = thisObject->globalObject();
    auto* prototype = globalObject->arrayBufferPrototype(mode);

    if (globalObject->arrayBufferSpeciesWatchpointSet(mode).state() == ClearWatchpoint) {
        dataLogLnIf(JSArrayBufferPrototypeInternal::verbose, "Initializing ArrayBuffer species watchpoints for ArrayBuffer.prototype: ", pointerDump(prototype), " with structure: ", pointerDump(prototype->structure()), "\nand ArrayBuffer: ", pointerDump(globalObject->arrayBufferConstructor(mode)), " with structure: ", pointerDump(globalObject->arrayBufferConstructor(mode)->structure()));
        globalObject->tryInstallArrayBufferSpeciesWatchpoint(mode);
        ASSERT(globalObject->arrayBufferSpeciesWatchpointSet(mode).state() != ClearWatchpoint);
    }

    return !thisObject->hasCustomProperties()
        && prototype == thisObject->getPrototypeDirect()
        && globalObject->arrayBufferSpeciesWatchpointSet(mode).state() == IsWatched;
}

ALWAYS_INLINE std::optional<JSValue> arrayBufferSpeciesConstructor(JSGlobalObject* globalObject, JSArrayBuffer* thisObject, ArrayBufferSharingMode mode)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    bool isValid = speciesWatchpointIsValid(thisObject, mode);
    scope.assertNoException();
    if (LIKELY(isValid))
        return std::nullopt;

    RELEASE_AND_RETURN(scope, arrayBufferSpeciesConstructorSlow(globalObject, thisObject, mode));
}

} // namespace JSC
