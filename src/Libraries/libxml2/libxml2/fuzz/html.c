/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 6, 2022.
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
#include <libxml/HTMLparser.h>
#include <libxml/HTMLtree.h>
#include <libxml/catalog.h>
#include "fuzz.h"

extern size_t LLVMFuzzerMutate(uint8_t *data, size_t size, size_t maxSize);
extern size_t LLVMFuzzerCustomMutator(uint8_t *data, size_t size, size_t maxSize, unsigned int seed);

size_t
LLVMFuzzerCustomMutator(uint8_t *data, size_t size, size_t maxSize, unsigned int seed) {
    xmlFuzzRndSetSeed(seed);

    const size_t optionsSize = sizeof(int);
    if (size < optionsSize)
        return LLVMFuzzerMutate(data, size, maxSize);

    // Mutate libxml2 parsing options in first byte of input (10% chance).
    if (xmlFuzzRnd() % 10 == 1)
        *((int *)&data[0]) = (int)xmlFuzzRnd();

    return optionsSize + LLVMFuzzerMutate(data + optionsSize, size - optionsSize, maxSize - optionsSize);
}

int
LLVMFuzzerInitialize(int *argc ATTRIBUTE_UNUSED,
                     char ***argv ATTRIBUTE_UNUSED) {
    xmlInitParser();
#ifdef LIBXML_CATALOG_ENABLED
    xmlInitializeCatalog();
#endif
    xmlSetGenericErrorFunc(NULL, xmlFuzzErrorFunc);

    return 0;
}

int
LLVMFuzzerTestOneInput(const char *data, size_t size) {
    static const size_t maxChunkSize = 128;
    xmlCharEncoding encoding;
    htmlDocPtr doc;
    htmlParserCtxtPtr ctxt;
    xmlOutputBufferPtr out;
    const char *docBuffer;
    size_t docSize, consumed, chunkSize;
    int opts, outSize;

    xmlFuzzDataInit(data, size);
    encoding = (xmlCharEncoding)(xmlFuzzDataHash() % 23); /* See <libxml/encoding.h>. */
    opts = xmlFuzzReadInt() | XML_PARSE_NONET;

    docBuffer = xmlFuzzReadRemaining(&docSize);
    if (docBuffer == NULL) {
        xmlFuzzDataCleanup();
        return(0);
    }

    /* Pull parser */

    /* Recovery mode allows more input to be fuzzed. */
    doc = htmlReadMemory(docBuffer, docSize, NULL, xmlGetCharEncodingName(encoding), opts | XML_PARSE_RECOVER);

    /*
     * Also test the serializer. Call htmlDocContentDumpOutput with our
     * own buffer to avoid encoding the output. The HTML encoding is
     * excruciatingly slow (see htmlEntityValueLookup).
     */
    out = xmlAllocOutputBuffer(NULL);
    htmlDocContentDumpOutput(out, doc, NULL);
    xmlOutputBufferClose(out);

    xmlFreeDoc(doc);

    /* Push parser */

    ctxt = htmlCreatePushParserCtxt(NULL, NULL, NULL, 0, NULL, encoding);
    htmlCtxtUseOptions(ctxt, opts);

    for (consumed = 0; consumed < docSize; consumed += chunkSize) {
        chunkSize = docSize - consumed;
        if (chunkSize > maxChunkSize)
            chunkSize = maxChunkSize;
        htmlParseChunk(ctxt, docBuffer + consumed, chunkSize, 0);
    }

    htmlParseChunk(ctxt, NULL, 0, 1);
    xmlFreeDoc(ctxt->myDoc);
    htmlFreeParserCtxt(ctxt);

    /* Cleanup */

    xmlFuzzDataCleanup();
    xmlResetLastError();

    return(0);
}

