/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 6, 2025.
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

#if defined(LIBXML_READER_ENABLED) && defined(LIBXML_PATTERN_ENABLED) && defined(LIBXML_OUTPUT_ENABLED)


/**
 * streamFile:
 * @filename: the file name to parse
 *
 * Parse and print information about an XML file.
 *
 * Returns the resulting doc with just the elements preserved.
 */
static xmlDocPtr
extractFile(const char *filename, const xmlChar *pattern) {
    xmlDocPtr doc;
    xmlTextReaderPtr reader;
    int ret;

    /*
     * build an xmlReader for that file
     */
    reader = xmlReaderForFile(filename, NULL, 0);
    if (reader != NULL) {
        /*
	 * add the pattern to preserve
	 */
        if (xmlTextReaderPreservePattern(reader, pattern, NULL) < 0) {
            fprintf(stderr, "%s : failed add preserve pattern %s\n",
	            filename, (const char *) pattern);
	}
	/*
	 * Parse and traverse the tree, collecting the nodes in the process
	 */
        ret = xmlTextReaderRead(reader);
        while (ret == 1) {
            ret = xmlTextReaderRead(reader);
        }
        if (ret != 0) {
            fprintf(stderr, "%s : failed to parse\n", filename);
	    xmlFreeTextReader(reader);
	    return(NULL);
        }
	/*
	 * get the resulting nodes
	 */
	doc = xmlTextReaderCurrentDoc(reader);
	/*
	 * Free up the reader
	 */
        xmlFreeTextReader(reader);
    } else {
        fprintf(stderr, "Unable to open %s\n", filename);
	return(NULL);
    }
    return(doc);
}

int main(int argc, char **argv) {
    const char *filename = "test3.xml";
    const char *pattern = "preserved";
    xmlDocPtr doc;

    if (argc == 3) {
        filename = argv[1];
	pattern = argv[2];
    }

    /*
     * this initialize the library and check potential ABI mismatches
     * between the version it was compiled for and the actual shared
     * library used.
     */
    LIBXML_TEST_VERSION

    doc = extractFile(filename, (const xmlChar *) pattern);
    if (doc != NULL) {
        /*
	 * output the result.
	 */
        xmlDocDump(stdout, doc);
	/*
	 * don't forget to free up the doc
	 */
	xmlFreeDoc(doc);
    }


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
    fprintf(stderr, "Reader, Pattern or output support not compiled in\n");
    exit(1);
}
#endif
