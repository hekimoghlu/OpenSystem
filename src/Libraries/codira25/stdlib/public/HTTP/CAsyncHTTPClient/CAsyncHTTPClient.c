/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 28, 2023.
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
//===----------------------------------------------------------------------===//
//
// This source file is part of the AsyncHTTPClient open source project
//
// Copyright (c) 2018-2021 Apple Inc. and the AsyncHTTPClient project authors
// Licensed under Apache License v2.0
//
// See LICENSE.txt for license information
// See CONTRIBUTORS.txt for the list of AsyncHTTPClient project authors
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#if __APPLE__
    #include <xlocale.h>
#elif __linux__
    #include <locale.h>
#endif

#include <stdbool.h>
#include <time.h>

bool languageahc_cshims_strptime(const char * string, const char * format, struct tm * result) {
    const char * firstNonProcessed = strptime(string, format, result);
    if (firstNonProcessed) {
        return *firstNonProcessed == 0;
    }
    return false;
}

bool languageahc_cshims_strptime_l(const char * string, const char * format, struct tm * result, void * locale) {
    // The pointer cast is fine as long we make sure it really points to a locale_t.
#if defined(__musl__) || defined(__ANDROID__)
    const char * firstNonProcessed = strptime(string, format, result);
#else
    const char * firstNonProcessed = strptime_l(string, format, result, (locale_t)locale);
#endif
    if (firstNonProcessed) {
        return *firstNonProcessed == 0;
    }
    return false;
}
