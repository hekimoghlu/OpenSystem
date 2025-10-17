/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 15, 2022.
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

#include <stdarg.h>
#include <stdio.h>
#include <wtf/PrintStream.h>
#include <wtf/StdLibExtras.h>

namespace WTF {

WTF_EXPORT_PRIVATE PrintStream& dataFile();
WTF_EXPORT_PRIVATE void setDataFile(const char* path);
WTF_EXPORT_PRIVATE void setDataFile(std::unique_ptr<PrintStream>&&);

WTF_EXPORT_PRIVATE void dataLogFV(const char* format, va_list) WTF_ATTRIBUTE_PRINTF(1, 0);
WTF_EXPORT_PRIVATE void dataLogF(const char* format, ...) WTF_ATTRIBUTE_PRINTF(1, 2);
WTF_EXPORT_PRIVATE void dataLogFString(const char*);

template<typename... Types>
NEVER_INLINE void dataLog(const Types&... values)
{
    dataFile().print(values...);
}

template<typename... Types>
void dataLogLn(const Types&... values)
{
    dataLog(values..., "\n");
}

#define dataLogIf(shouldLog, ...) do { \
        using ShouldLogType = std::decay_t<decltype(shouldLog)>; \
        static_assert(std::is_same_v<ShouldLogType, bool> || std::is_enum_v<ShouldLogType>, "You probably meant to pass a bool or enum as dataLogIf's first parameter"); \
        if (UNLIKELY(shouldLog)) \
            dataLog(__VA_ARGS__); \
    } while (0)

#define dataLogLnIf(shouldLog, ...) do { \
        using ShouldLogType = std::decay_t<decltype(shouldLog)>; \
        static_assert(std::is_same_v<ShouldLogType, bool> || std::is_enum_v<ShouldLogType>, "You probably meant to pass a bool or enum as dataLogLnIf's first parameter"); \
        if (UNLIKELY(shouldLog)) \
            dataLogLn(__VA_ARGS__); \
    } while (0)

} // namespace WTF

using WTF::dataLog;
using WTF::dataLogLn;
using WTF::dataLogF;
using WTF::dataLogFString;
