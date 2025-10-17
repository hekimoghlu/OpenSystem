/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 17, 2022.
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

#include "compiler/translator/Diagnostics.h"

#include "common/debug.h"
#include "compiler/preprocessor/SourceLocation.h"
#include "compiler/translator/Common.h"
#include "compiler/translator/InfoSink.h"

namespace sh
{

TDiagnostics::TDiagnostics(TInfoSinkBase &infoSink)
    : mInfoSink(infoSink), mNumErrors(0), mNumWarnings(0)
{}

TDiagnostics::~TDiagnostics() {}

void TDiagnostics::writeInfo(Severity severity,
                             const angle::pp::SourceLocation &loc,
                             const char *reason,
                             const char *token)
{
    switch (severity)
    {
        case SH_ERROR:
            ++mNumErrors;
            break;
        case SH_WARNING:
            ++mNumWarnings;
            break;
        default:
            UNREACHABLE();
            break;
    }

    /* VC++ format: file(linenum) : error #: 'token' : extrainfo */
    mInfoSink.prefix(severity);
    mInfoSink.location(loc.file, loc.line);
    mInfoSink << "'" << token << "' : " << reason << "\n";
}

void TDiagnostics::globalError(const char *message)
{
    ++mNumErrors;
    mInfoSink.prefix(SH_ERROR);
    mInfoSink << message << "\n";
}

void TDiagnostics::error(const angle::pp::SourceLocation &loc,
                         const char *reason,
                         const char *token)
{
    writeInfo(SH_ERROR, loc, reason, token);
}

void TDiagnostics::warning(const angle::pp::SourceLocation &loc,
                           const char *reason,
                           const char *token)
{
    writeInfo(SH_WARNING, loc, reason, token);
}

void TDiagnostics::error(const TSourceLoc &loc, const char *reason, const char *token)
{
    angle::pp::SourceLocation srcLoc;
    srcLoc.file = loc.first_file;
    srcLoc.line = loc.first_line;
    error(srcLoc, reason, token);
}

void TDiagnostics::warning(const TSourceLoc &loc, const char *reason, const char *token)
{
    angle::pp::SourceLocation srcLoc;
    srcLoc.file = loc.first_file;
    srcLoc.line = loc.first_line;
    warning(srcLoc, reason, token);
}

void TDiagnostics::print(ID id, const angle::pp::SourceLocation &loc, const std::string &text)
{
    writeInfo(isError(id) ? SH_ERROR : SH_WARNING, loc, message(id), text.c_str());
}

void TDiagnostics::resetErrorCount()
{
    mNumErrors   = 0;
    mNumWarnings = 0;
}

PerformanceDiagnostics::PerformanceDiagnostics(TDiagnostics *diagnostics)
    : mDiagnostics(diagnostics)
{
    ASSERT(diagnostics);
}

void PerformanceDiagnostics::warning(const TSourceLoc &loc, const char *reason, const char *token)
{
    mDiagnostics->warning(loc, reason, token);
}

}  // namespace sh
