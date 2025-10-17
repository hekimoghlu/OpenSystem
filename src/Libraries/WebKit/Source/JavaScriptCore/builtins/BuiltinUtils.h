/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 1, 2025.
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

#include "ConstructAbility.h"
#include "ConstructorKind.h"
#include "ImplementationVisibility.h"
#include "InlineAttribute.h"

namespace JSC {

#define INITIALIZE_BUILTIN_NAMES(name) , m_##name(JSC::Identifier::fromString(vm, #name ""_s)), m_##name##PrivateName(JSC::Identifier::fromUid(JSC::PrivateName(JSC::PrivateName::PrivateSymbol, #name ""_s)))
#define DECLARE_BUILTIN_NAMES(name) const JSC::Identifier m_##name; const JSC::Identifier m_##name##PrivateName;
#define DECLARE_BUILTIN_IDENTIFIER_ACCESSOR(name) \
    const JSC::Identifier& name##PublicName() const { return m_##name; } \
    const JSC::Identifier& name##PrivateName() const { return m_##name##PrivateName; }

class Identifier;
class SourceCode;
class UnlinkedFunctionExecutable;
class VM;

JS_EXPORT_PRIVATE UnlinkedFunctionExecutable* createBuiltinExecutable(VM&, const SourceCode&, const Identifier&, ImplementationVisibility, ConstructorKind, ConstructAbility, InlineAttribute);
    
} // namespace JSC
