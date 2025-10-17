/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 29, 2021.
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

// Â© 2019 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html

#include <fstream>
#include <iostream>
#include <sstream>
#include <stddef.h>
#include <stdint.h>
#include <string>

#include "cmemory.h"

#if APPLE_ICU_CHANGES
// rdar://70810661 (ability to compile ICU with asan and libfuzzer)
#elif defined(__has_feature) && __has_feature(coverage_sanitizer))

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size);

int main(int argc, char* argv[])
{
    bool show_warning = true;
    bool show_error = true;
#if UPRV_HAS_FEATURE(address_sanitizer)
    show_warning = false;
#endif
#if UPRV_HAS_FEATURE(memory_sanitizer)
    show_warning = false;
#endif
    if (argc > 2 && strcmp(argv[2], "-q") == 0) {
        show_warning = false;
        show_error = false;
    }
    if (show_warning) {
        std::cerr << "WARNING: This binary work only under build configure with" << std::endl
                  << " CFLAGS=\"-fsanitize=$SANITIZE\""
                  << " CXXFLAGS=\"-fsanitize=$SANITIZE\""
                  << " ./runConfigureICU ... " << std::endl
                  << "  where $SANITIZE is 'address' or 'memory'" << std::endl
                  << "Please run the above step and make tests to rebuild" << std::endl;
        // Do not return -1 here so we will pass the unit test.
    }
    if (argc < 2) {
        if (show_error) {
            std::cerr << "Usage: " << argv[0] << "  testcasefile [-q]" << std::endl
                      << "  -q : quiet while error" << std::endl;
        }
        return -1;
    }
    const char *path = argv[1];
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        if (show_error) {
            std::cerr << "Cannot open testcase file " << path << std::endl;
        }
        return -1;
    }
    std::ostringstream ostrm;
    ostrm << file.rdbuf();
    LLVMFuzzerTestOneInput(reinterpret_cast<const uint8_t*>(ostrm.str().c_str()), ostrm.str().size());

    return 0;
}

#endif  // APPLE_ICU_CHANGES
