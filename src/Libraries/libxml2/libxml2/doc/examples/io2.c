/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 22, 2021.
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
#include <libxml/parser.h>

#if defined(LIBXML_TREE_ENABLED) && defined(LIBXML_OUTPUT_ENABLED)
int
main(void)
{

    xmlNodePtr n;
    xmlDocPtr doc;
    xmlChar *xmlbuff;
    int buffersize;

    /*
     * Create the document.
     */
    doc = xmlNewDoc(BAD_CAST "1.0");
    n = xmlNewNode(NULL, BAD_CAST "root");
    xmlNodeSetContent(n, BAD_CAST "content");
    xmlDocSetRootElement(doc, n);

    /*
     * Dump the document to a buffer and print it
     * for demonstration purposes.
     */
    xmlDocDumpFormatMemory(doc, &xmlbuff, &buffersize, 1);
    printf("%s", (char *) xmlbuff);

    /*
     * Free associated memory.
     */
    xmlFree(xmlbuff);
    xmlFreeDoc(doc);

    return (0);

}
#else
#include <stdio.h>

int
main(void)
{
    fprintf(stderr,
            "library not configured with tree and output support\n");
    return (1);
}
#endif
