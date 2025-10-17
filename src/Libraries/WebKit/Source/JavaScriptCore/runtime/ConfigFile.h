/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 17, 2021.
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

#include <limits.h>

namespace JSC {

class ConfigFile {
public:
    JS_EXPORT_PRIVATE ConfigFile(const char*);

    JS_EXPORT_PRIVATE static void setProcessName(const char*);
    JS_EXPORT_PRIVATE static void setParentProcessName(const char*);
    JS_EXPORT_PRIVATE void parse();

private:
    void canonicalizePaths();

#if OS(WINDOWS)
    static const size_t s_maxPathLength = 260; // Windows value for "MAX_PATH"
#elif defined(PATH_MAX)
    static const size_t s_maxPathLength = PATH_MAX;
#else
    static const size_t s_maxPathLength = 4095;
#endif

    static char s_processName[];
    static char s_parentProcessName[];

    char m_filename[s_maxPathLength + 1];
    char m_configDirectory[s_maxPathLength + 1];
};

JS_EXPORT_PRIVATE void processConfigFile(const char* configFile, const char* processName, const char* parentProcessName = nullptr);

} // namespace JSC
