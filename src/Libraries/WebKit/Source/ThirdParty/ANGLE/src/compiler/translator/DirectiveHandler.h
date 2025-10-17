/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 4, 2025.
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

#ifndef COMPILER_TRANSLATOR_DIRECTIVEHANDLER_H_
#define COMPILER_TRANSLATOR_DIRECTIVEHANDLER_H_

#include "GLSLANG/ShaderLang.h"
#include "common/angleutils.h"
#include "compiler/preprocessor/DirectiveHandlerBase.h"
#include "compiler/preprocessor/Macro.h"
#include "compiler/translator/ExtensionBehavior.h"
#include "compiler/translator/Pragma.h"

namespace sh
{
class TDiagnostics;

class TDirectiveHandler : public angle::pp::DirectiveHandler, angle::NonCopyable
{
  public:
    TDirectiveHandler(TExtensionBehavior &extBehavior,
                      TDiagnostics &diagnostics,
                      int &shaderVersion,
                      sh::GLenum shaderType);
    ~TDirectiveHandler() override;

    const TPragma &pragma() const { return mPragma; }
    const TExtensionBehavior &extensionBehavior() const { return mExtensionBehavior; }

    void handleError(const angle::pp::SourceLocation &loc, const std::string &msg) override;

    void handlePragma(const angle::pp::SourceLocation &loc,
                      const std::string &name,
                      const std::string &value,
                      bool stdgl) override;

    void handleExtension(const angle::pp::SourceLocation &loc,
                         const std::string &name,
                         const std::string &behavior) override;

    void handleVersion(const angle::pp::SourceLocation &loc,
                       int version,
                       ShShaderSpec spec,
                       angle::pp::MacroSet *macro_set) override;

  private:
    TPragma mPragma;
    TExtensionBehavior &mExtensionBehavior;
    TDiagnostics &mDiagnostics;
    int &mShaderVersion;
    sh::GLenum mShaderType;
};

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_DIRECTIVEHANDLER_H_
