/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 27, 2024.
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

#include "Yarr.h"
#include <optional>
#include <wtf/Forward.h>

namespace JSC { namespace Yarr {

struct CharacterClass;
struct ClassSet;
enum class CompileMode : uint8_t;

JS_EXPORT_PRIVATE std::optional<BuiltInCharacterClassID> unicodeMatchPropertyValue(WTF::String, WTF::String);
JS_EXPORT_PRIVATE std::optional<BuiltInCharacterClassID> unicodeMatchProperty(WTF::String, CompileMode);

std::unique_ptr<CharacterClass> createUnicodeCharacterClassFor(BuiltInCharacterClassID);
JS_EXPORT_PRIVATE bool characterClassMayContainStrings(BuiltInCharacterClassID unicodeClassID);

} } // namespace JSC::Yarr
