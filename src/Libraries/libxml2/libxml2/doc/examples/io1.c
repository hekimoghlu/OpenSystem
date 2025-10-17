/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 26, 2022.
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
#include <string.h>
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <libxml/xinclude.h>
#include <libxml/xmlIO.h>

#ifdef LIBXML_XINCLUDE_ENABLED
static const char *result = "<list><people>a</people><people>b</people></list>";
static const char *cur = NULL;
static int rlen;

/**
 * sqlMatch:
 * @URI: an URI to test
 *
 * Check for an sql: query
 *
 * Returns 1 if yes and 0 if another Input module should be used
 */
static int
sqlMatch(const char * URI) {
    if ((URI != NULL) && (!strncmp(URI, "sql:", 4)))
        return(1);
    return(0);
}

/**
 * sqlOpen:
 * @URI: an URI to test
 *
 * Return a pointer to the sql: query handler, in this example simply
 * the current pointer...
 *
 * Returns an Input context or NULL in case or error
 */
static void *
sqlOpen(const char * URI) {
    if ((URI == NULL) || (strncmp(URI, "sql:", 4)))
        return(NULL);
    cur = result;
    rlen = strlen(result);
    return((void *) cur);
}

/**
 * sqlClose:
 * @context: the read context
 *
 * Close the sql: query handler
 *
 * Returns 0 or -1 in case of error
 */
static int
sqlClose(void * context) {
    if (context == NULL) return(-1);
    cur = NULL;
    rlen = 0;
    return(0);
}

/**
 * sqlRead:
 * @context: the read context
 * @buffer: where to store data
 * @len: number of bytes to read
 *
 * Implement an sql: query read.
 *
 * Returns the number of bytes read or -1 in case of error
 */
static int
sqlRead(void * context, char * buffer, int len) {
   const char *ptr = (const char *) context;

   if ((context == NULL) || (buffer == NULL) || (len < 0))
       return(-1);

   if (len > rlen) len = rlen;
   memcpy(buffer, ptr, len);
   rlen -= len;
   return(len);
}

const char *include = "<?xml version='1.0'?>\n\
<document xmlns:xi=\"http://www.w3.org/2003/XInclude\">\n\
  <p>List of people:</p>\n\
  <xi:include href=\"sql:select_name_from_people\"/>\n\
</document>\n";

int main(void) {
    xmlDocPtr doc;

    /*
     * this initialize the library and check potential ABI mismatches
     * between the version it was compiled for and the actual shared
     * library used.
     */
    LIBXML_TEST_VERSION

    /*
     * register the new I/O handlers
     */
    if (xmlRegisterInputCallbacks(sqlMatch, sqlOpen, sqlRead, sqlClose) < 0) {
        fprintf(stderr, "failed to register SQL handler\n");
	exit(1);
    }
    /*
     * parse include into a document
     */
    doc = xmlReadMemory(include, strlen(include), "include.xml", NULL, 0);
    if (doc == NULL) {
        fprintf(stderr, "failed to parse the including file\n");
	exit(1);
    }

    /*
     * apply the XInclude process, this should trigger the I/O just
     * registered.
     */
    if (xmlXIncludeProcess(doc) <= 0) {
        fprintf(stderr, "XInclude processing failed\n");
	exit(1);
    }

#ifdef LIBXML_OUTPUT_ENABLED
    /*
     * save the output for checking to stdout
     */
    xmlDocDump(stdout, doc);
#endif

    /*
     * Free the document
     */
    xmlFreeDoc(doc);

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
    fprintf(stderr, "XInclude support not compiled in\n");
    exit(1);
}
#endif
