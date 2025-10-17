/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 24, 2023.
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
#include "fuzz.h"

int
LLVMFuzzerInitialize(int *argc_p, char ***argv_p) {
    const char *dir = getenv("STATIC_XML_FILE_DIR");
    if (dir && dir[0] == '\0')
        dir = NULL;
    return xsltFuzzXPathInit(argc_p, argv_p, dir);
}

int
LLVMFuzzerTestOneInput(const char *data, size_t size) {
    xmlXPathObjectPtr xpathObj = xsltFuzzXPath(data, size);
    xsltFuzzXPathFreeObject(xpathObj);

    return 0;
}
