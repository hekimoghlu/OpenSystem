/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 24, 2024.
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

//
// Copyright 2011 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/preprocessor/Macro.h"

#include "common/angleutils.h"
#include "compiler/preprocessor/Token.h"

namespace angle
{

namespace pp
{

Macro::Macro() : predefined(false), disabled(false), expansionCount(0), type(kTypeObj) {}

Macro::~Macro() {}

bool Macro::equals(const Macro &other) const
{
    return (type == other.type) && (name == other.name) && (parameters == other.parameters) &&
           (replacements == other.replacements);
}

void PredefineMacro(MacroSet *macroSet, const char *name, int value)
{
    Token token;
    token.type = Token::CONST_INT;
    token.text = ToString(value);

    std::shared_ptr<Macro> macro = std::make_shared<Macro>();
    macro->predefined            = true;
    macro->type                  = Macro::kTypeObj;
    macro->name                  = name;
    macro->replacements.push_back(token);

    (*macroSet)[name] = macro;
}

}  // namespace pp

}  // namespace angle
