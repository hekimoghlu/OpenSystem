/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 26, 2023.
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

<![CDATA[
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <libxml/xmlmemory.h>
#include <libxml/parser.h>


xmlDocPtr
parseDoc(char *docname, char *uri) {

	xmlDocPtr doc;
	xmlNodePtr cur;
	xmlNodePtr newnode;
	xmlAttrPtr newattr;

	doc = xmlParseFile(docname);
	
	if (doc == NULL ) {
		fprintf(stderr,"Document not parsed successfully. \n");
		return (NULL);
	}
	
	cur = xmlDocGetRootElement(doc);
	
	if (cur == NULL) {
		fprintf(stderr,"empty document\n");
		xmlFreeDoc(doc);
		return (NULL);
	}
	
	if (xmlStrcmp(cur->name, (const xmlChar *) "story")) {
		fprintf(stderr,"document of the wrong type, root node != story");
		xmlFreeDoc(doc);
		return (NULL);
	}
	
	newnode = xmlNewTextChild (cur, NULL, "reference", NULL);
	newattr = xmlNewProp (newnode, "uri", uri);
	return(doc);
}

int
main(int argc, char **argv) {

	char *docname;
	char *uri;
	xmlDocPtr doc;

	if (argc <= 2) {
		printf("Usage: %s docname, uri\n", argv[0]);
		return(0);
	}

	docname = argv[1];
	uri = argv[2];
	doc = parseDoc (docname, uri);
	if (doc != NULL) {
		xmlSaveFormatFile (docname, doc, 1);
		xmlFreeDoc(doc);
	}
	return (1);
}
]]>
