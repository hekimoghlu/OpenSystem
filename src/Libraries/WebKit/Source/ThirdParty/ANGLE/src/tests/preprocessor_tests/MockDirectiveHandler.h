/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 20, 2023.
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
// Copyright 2012 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef PREPROCESSOR_TESTS_MOCK_DIRECTIVE_HANDLER_H_
#define PREPROCESSOR_TESTS_MOCK_DIRECTIVE_HANDLER_H_

#include "compiler/preprocessor/DirectiveHandlerBase.h"
#include "gmock/gmock.h"

namespace angle
{

class MockDirectiveHandler : public pp::DirectiveHandler
{
  public:
    MOCK_METHOD2(handleError, void(const pp::SourceLocation &loc, const std::string &msg));

    MOCK_METHOD4(handlePragma,
                 void(const pp::SourceLocation &loc,
                      const std::string &name,
                      const std::string &value,
                      bool stdgl));

    MOCK_METHOD3(handleExtension,
                 void(const pp::SourceLocation &loc,
                      const std::string &name,
                      const std::string &behavior));

    MOCK_METHOD4(handleVersion,
                 void(const pp::SourceLocation &loc,
                      int version,
                      ShShaderSpec spec,
                      pp::MacroSet *macro_set));
};

}  // namespace angle

#endif  // PREPROCESSOR_TESTS_MOCK_DIRECTIVE_HANDLER_H_
