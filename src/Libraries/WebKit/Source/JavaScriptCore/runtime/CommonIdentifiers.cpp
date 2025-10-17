/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 6, 2023.
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
#include "config.h"
#include "CommonIdentifiers.h"

#include "BuiltinNames.h"
#include "IdentifierInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CommonIdentifiers);

#define INITIALIZE_PROPERTY_NAME(name) , name(Identifier::fromString(vm, #name ""_s))
#define INITIALIZE_KEYWORD(name) , name##Keyword(Identifier::fromString(vm, #name ""_s))
#define INITIALIZE_PRIVATE_NAME(name) , name##PrivateName(m_builtinNames->name##PrivateName())
#define INITIALIZE_SYMBOL(name) , name##Symbol(m_builtinNames->name##Symbol())
#define INITIALIZE_PRIVATE_FIELD_NAME(name) , name##PrivateField(Identifier::fromString(vm, "#" #name ""_s))

CommonIdentifiers::CommonIdentifiers(VM& vm)
    : nullIdentifier()
    , emptyIdentifier(Identifier::EmptyIdentifier)
    , underscoreProto(Identifier::fromString(vm, "__proto__"_s))
    , useStrictIdentifier(Identifier::fromString(vm, "use strict"_s))
    , timesIdentifier(Identifier::fromString(vm, "*"_s))
    , negativeOneIdentifier(Identifier::fromString(vm, "-1"_s))
    , m_builtinNames(makeUnique<BuiltinNames>(vm, this))
    JSC_PARSER_PRIVATE_NAMES(INITIALIZE_PRIVATE_NAME)
    JSC_COMMON_IDENTIFIERS_EACH_KEYWORD(INITIALIZE_KEYWORD)
    JSC_COMMON_IDENTIFIERS_EACH_PROPERTY_NAME(INITIALIZE_PROPERTY_NAME)
    JSC_COMMON_PRIVATE_IDENTIFIERS_EACH_WELL_KNOWN_SYMBOL(INITIALIZE_SYMBOL)
    JSC_COMMON_IDENTIFIERS_EACH_PRIVATE_FIELD(INITIALIZE_PRIVATE_FIELD_NAME)
{
}

CommonIdentifiers::~CommonIdentifiers() = default;

void CommonIdentifiers::appendExternalName(const Identifier& publicName, const Identifier& privateName)
{
    m_builtinNames->appendExternalName(publicName, privateName);
}

} // namespace JSC
