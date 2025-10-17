/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 13, 2024.
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

// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef SUPPORT_SET_WINDOWS_CRT_REPORT_MODE_H
#define SUPPORT_SET_WINDOWS_CRT_REPORT_MODE_H

#ifndef _DEBUG
#  error _DEBUG must be defined when using this header
#endif

#ifndef _WIN32
#  error This header can only be used when targeting Windows
#endif

#include <crtdbg.h>

// On Windows in debug builds the default assertion handler opens a new dialog
// window which must be dismissed manually by the user. This function overrides
// that setting and instead changes the assertion handler to log to stderr
// instead.
inline int init_crt_report_mode()
{
  _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_DEBUG);
  _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
  _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_DEBUG);
  return 0;
}

static int init_crt_anchor = init_crt_report_mode();

#endif // SUPPORT_SET_WINDOWS_CRT_REPORT_MODE_H
