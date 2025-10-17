/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 8, 2023.
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
#include <stdio.h>
#include <libxml/xmlreader.h>

#ifdef LIBXML_READER_ENABLED

static void processDoc(xmlTextReaderPtr readerPtr) {
    int ret;
    xmlDocPtr docPtr;
    const xmlChar *URL;

    ret = xmlTextReaderRead(readerPtr);
    while (ret == 1) {
      ret = xmlTextReaderRead(readerPtr);
    }

    /*
     * One can obtain the document pointer to get interesting
     * information about the document like the URL, but one must also
     * be sure to clean it up at the end (see below).
     */
    docPtr = xmlTextReaderCurrentDoc(readerPtr);
    if (NULL == docPtr) {
      fprintf(stderr, "failed to obtain document\n");      
      return;
    }
      
    URL = docPtr->URL;
    if (NULL == URL) {
      fprintf(stderr, "Failed to obtain URL\n");      
    }

    if (ret != 0) {
      fprintf(stderr, "%s: Failed to parse\n", URL);
      return;
    }

    printf("%s: Processed ok\n", (const char *)URL);
}

int main(int argc, char **argv) {
    xmlTextReaderPtr readerPtr;
    int i;
    xmlDocPtr docPtr;

    if (argc < 2)
        return(1);

    /*
     * this initialises the library and check potential ABI mismatches
     * between the version it was compiled for and the actual shared
     * library used.
     */
    LIBXML_TEST_VERSION

    /*
     * Create a new reader for the first file and process the
     * document.
     */
    readerPtr = xmlReaderForFile(argv[1], NULL, 0);
    if (NULL == readerPtr) {
      fprintf(stderr, "%s: failed to create reader\n", argv[1]);      
      return(1);
    }
    processDoc(readerPtr);

    /*
     * The reader can be reused for subsequent files.
     */
    for (i=2; i < argc; ++i) {
      	xmlReaderNewFile(readerPtr, argv[i], NULL, 0);
	if (NULL == readerPtr) {
	  fprintf(stderr, "%s: failed to create reader\n", argv[i]);      
	  return(1);
	}
        processDoc(readerPtr);
    }

    /*
     * Since we've called xmlTextReaderCurrentDoc, we now have to
     * clean up after ourselves.  We only have to do this the last
     * time, because xmlReaderNewFile calls xmlCtxtReset which takes
     * care of it.
     */
    docPtr = xmlTextReaderCurrentDoc(readerPtr);
    if (docPtr != NULL)
      xmlFreeDoc(docPtr);

    /*
     * Clean up the reader.
     */
    xmlFreeTextReader(readerPtr);

    /*
     * Cleanup function for the XML library.
     */
    xmlCleanupParser();
    /*
     * this is to debug memory for regression tests
     */
    xmlMemoryDump();
    return(0);
}

#else
int main(void) {
    fprintf(stderr, "xmlReader support not compiled in\n");
    exit(1);
}
#endif
