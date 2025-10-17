/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 3, 2021.
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
#ifndef mach_o_LoggingStub_h
#define mach_o_LoggingStub_h

#include <cstdarg>

#include "MachODefines.h"
#include "va_list_wrap.h"

namespace mach_o
{

using WarningHandler = void(*)(const void* context, const char* format, va_list_wrap);
void setWarningHandler(WarningHandler) VIS_HIDDEN;
bool hasWarningHandler() VIS_HIDDEN;

__attribute__((format(printf, 2, 3)))
void warning(const void* context, const char* format, ...) VIS_HIDDEN;
}

#endif // mach_o_LoggingStub_h
