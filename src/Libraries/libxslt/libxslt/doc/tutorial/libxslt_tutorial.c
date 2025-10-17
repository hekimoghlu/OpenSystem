/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 6, 2021.
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
#include <string.h>
#include <libxml/xmlmemory.h>
#include <libxml/debugXML.h>
#include <libxml/HTMLtree.h>
#include <libxml/xmlIO.h>
#include <libxml/DOCBparser.h>
#include <libxml/xinclude.h>
#include <libxml/catalog.h>
#include <libxslt/xslt.h>
#include <libxslt/xsltInternals.h>
#include <libxslt/transform.h>
#include <libxslt/xsltutils.h>



extern int xmlLoadExtDtdDefaultValue;

static void usage(const char *name) {
    printf("Usage: %s [options] stylesheet file [file ...]\n", name);
    printf("      --param name value : pass a (parameter,value) pair\n");

}

int
main(int argc, char **argv) {
	int i;
	const char *params[16 + 1];
	int nbparams = 0;
	xsltStylesheetPtr cur = NULL;
	xmlDocPtr doc, res;

	if (argc <= 1) {
		usage(argv[0]);
		return(1);
	}
	

 for (i = 1; i < argc; i++) {
        if (argv[i][0] != '-')
            break;
	if ((!strcmp(argv[i], "-param")) ||
                   (!strcmp(argv[i], "--param"))) {
		i++;
		params[nbparams++] = argv[i++];
		params[nbparams++] = argv[i];
		if (nbparams >= 16) {
			fprintf(stderr, "too many params\n");
			return (1);
		}
        }  else {
            fprintf(stderr, "Unknown option %s\n", argv[i]);
            usage(argv[0]);
            return (1);
        }
    }

	params[nbparams] = NULL;
	xmlSubstituteEntitiesDefault(1);
	xmlLoadExtDtdDefaultValue = 1;
	cur = xsltParseStylesheetFile((const xmlChar *)argv[i]);
	i++;
	doc = xmlParseFile(argv[i]);
	res = xsltApplyStylesheet(cur, doc, params);
	xsltSaveResultToFile(stdout, res, cur);

	xsltFreeStylesheet(cur);
	xmlFreeDoc(res);
	xmlFreeDoc(doc);

        xsltCleanupGlobals();
        xmlCleanupParser();
	return(0);

}
