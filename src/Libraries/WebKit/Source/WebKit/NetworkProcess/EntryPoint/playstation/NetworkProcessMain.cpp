/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 4, 2023.
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
#include "config.h"
#include "NetworkProcessMain.h"

#include <dlfcn.h>
#include <process-initialization/nk-networkprocess.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static void loadLibraryOrExit(const char* name)
{
    if (!dlopen(name, RTLD_NOW)) {
        fprintf(stderr, "Failed to load %s.\n", name);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Unexpected argument count %d\n", argc);
        exit(EXIT_FAILURE);
    }

    loadLibraryOrExit(OpenSSL_LOAD_AT);
#if defined(Brotli_LOAD_AT)
    loadLibraryOrExit(Brotli_LOAD_AT);
#endif
    loadLibraryOrExit(CURL_LOAD_AT);
    loadLibraryOrExit(ICU_LOAD_AT);
#if defined(LibPSL_LOAD_AT)
    loadLibraryOrExit(LibPSL_LOAD_AT);
#endif
#if defined(SQLite_LOAD_AT)
    loadLibraryOrExit(SQLite_LOAD_AT);
#endif
    loadLibraryOrExit(WebKitRequirements_LOAD_AT);
#if !ENABLE(STATIC_JSC)
    loadLibraryOrExit("libJavaScriptCore");
#endif
    loadLibraryOrExit("libWebKit");

    char* coreProcessIdentifier = argv[1];

    char connectionIdentifier[16];
    snprintf(connectionIdentifier, sizeof(connectionIdentifier), "%d", PlayStation::getConnectionIdentifier());

    char program[] = "dummy";
    char* internalArgv[] = {
        program,
        coreProcessIdentifier,
        connectionIdentifier,
        0
    };
    return WebKit::NetworkProcessMain(sizeof(internalArgv) / sizeof(char*), internalArgv);
}
