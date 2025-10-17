/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 24, 2024.
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
#include <libxml/uri.h>
#include "fuzz.h"

int
LLVMFuzzerTestOneInput(const char *data, size_t size) {
    xmlURIPtr uri;
    char *str[2] = { NULL, NULL };
    size_t numStrings;

    if (size > 10000)
        return(0);

    numStrings = xmlFuzzExtractStrings(data, size, str, 2);

    uri = xmlParseURI(str[0]);
    xmlFree(xmlSaveUri(uri));
    xmlFreeURI(uri);

    uri = xmlParseURIRaw(str[0], 1);
    xmlFree(xmlSaveUri(uri));
    xmlFreeURI(uri);

    xmlFree(xmlURIUnescapeString(str[0], -1, NULL));
    xmlFree(xmlURIEscape(BAD_CAST str[0]));
    xmlFree(xmlCanonicPath(BAD_CAST str[0]));
    xmlFree(xmlPathToURI(BAD_CAST str[0]));

    if (numStrings >= 2) {
        xmlFree(xmlBuildURI(BAD_CAST str[1], BAD_CAST str[0]));
        xmlFree(xmlBuildRelativeURI(BAD_CAST str[1], BAD_CAST str[0]));
        xmlFree(xmlURIEscapeStr(BAD_CAST str[0], BAD_CAST str[1]));
    }

    /* Modifies string, so must come last. */
    xmlNormalizeURIPath(str[0]);

    xmlFree(str[0]);
    xmlFree(str[1]);

    return 0;
}

