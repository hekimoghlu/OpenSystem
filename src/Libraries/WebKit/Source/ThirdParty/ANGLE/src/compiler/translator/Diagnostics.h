/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 23, 2023.
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

#ifndef COMPILER_TRANSLATOR_DIAGNOSTICS_H_
#define COMPILER_TRANSLATOR_DIAGNOSTICS_H_

#include "common/angleutils.h"
#include "compiler/preprocessor/DiagnosticsBase.h"
#include "compiler/translator/Severity.h"

namespace sh
{

class TInfoSinkBase;
struct TSourceLoc;

class TDiagnostics : public angle::pp::Diagnostics, angle::NonCopyable
{
  public:
    TDiagnostics(TInfoSinkBase &infoSink);
    ~TDiagnostics() override;

    int numErrors() const { return mNumErrors; }
    int numWarnings() const { return mNumWarnings; }

    void error(const angle::pp::SourceLocation &loc, const char *reason, const char *token);
    void warning(const angle::pp::SourceLocation &loc, const char *reason, const char *token);

    void error(const TSourceLoc &loc, const char *reason, const char *token);
    void warning(const TSourceLoc &loc, const char *reason, const char *token);

    void globalError(const char *message);

    void resetErrorCount();

  protected:
    void writeInfo(Severity severity,
                   const angle::pp::SourceLocation &loc,
                   const char *reason,
                   const char *token);

    void print(ID id, const angle::pp::SourceLocation &loc, const std::string &text) override;

  private:
    TInfoSinkBase &mInfoSink;
    int mNumErrors;
    int mNumWarnings;
};

// Diagnostics wrapper to use when the code is only allowed to generate warnings.
class PerformanceDiagnostics : public angle::NonCopyable
{
  public:
    PerformanceDiagnostics(TDiagnostics *diagnostics);

    void warning(const TSourceLoc &loc, const char *reason, const char *token);

  private:
    TDiagnostics *mDiagnostics;
};

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_DIAGNOSTICS_H_
