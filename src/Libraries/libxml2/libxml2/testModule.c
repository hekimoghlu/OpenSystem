/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 19, 2024.
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
#include "libxml.h"
#ifdef LIBXML_MODULES_ENABLED
#include <libxml/xmlversion.h>

#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#include <libxml/xmlmemory.h>
#include <libxml/debugXML.h>
#include <libxml/xmlmodule.h>

#ifdef _WIN32
#define MODULE_PATH "."
#include <stdlib.h> /* for _MAX_PATH */
#ifndef __MINGW32__
#define PATH_MAX _MAX_PATH
#endif
#else
#define MODULE_PATH ".libs"
#endif

/* Used for SCO Openserver*/
#ifndef PATH_MAX
#ifdef _POSIX_PATH_MAX
#define PATH_MAX _POSIX_PATH_MAX
#else
#define PATH_MAX 4096
#endif
#endif

typedef int (*hello_world_t)(void);

int main(int argc ATTRIBUTE_UNUSED, char **argv ATTRIBUTE_UNUSED) {
    xmlChar filename[PATH_MAX];
    xmlModulePtr module = NULL;
    hello_world_t hello_world = NULL;

    /* build the module filename, and confirm the module exists */
    xmlStrPrintf(filename, sizeof(filename),
                 "%s/testdso%s",
                 (const xmlChar*)MODULE_PATH,
		 (const xmlChar*)LIBXML_MODULE_EXTENSION);

    module = xmlModuleOpen((const char*)filename, 0);
    if (module)
      {
        if (xmlModuleSymbol(module, "hello_world", (void **) &hello_world)) {
	    fprintf(stderr, "Failure to lookup\n");
	    return(1);
	}
	if (hello_world == NULL) {
	    fprintf(stderr, "Lookup returned NULL\n");
	    return(1);
	}

        (*hello_world)();

        xmlModuleClose(module);
      }

    xmlMemoryDump();

    return(0);
}

#else
#include <stdio.h>
int main(int argc ATTRIBUTE_UNUSED, char **argv ATTRIBUTE_UNUSED) {
    printf("%s : Module support not compiled in\n", argv[0]);
    return(0);
}
#endif /* LIBXML_SCHEMAS_ENABLED */
