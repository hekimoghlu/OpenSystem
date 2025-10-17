/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 28, 2021.
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
#ifndef __XML_MODULE_H__
#define __XML_MODULE_H__

#include <libxml/xmlversion.h>

#ifdef LIBXML_MODULES_ENABLED

#ifdef __cplusplus
extern "C" {
#endif

/**
 * xmlModulePtr:
 *
 * A handle to a dynamically loaded module
 */
typedef struct _xmlModule xmlModule;
typedef xmlModule *xmlModulePtr;

/**
 * xmlModuleOption:
 *
 * enumeration of options that can be passed down to xmlModuleOpen()
 */
typedef enum {
    XML_MODULE_LAZY = 1,	/* lazy binding */
    XML_MODULE_LOCAL= 2		/* local binding */
} xmlModuleOption;

XMLPUBFUN xmlModulePtr XMLCALL xmlModuleOpen	(const char *filename,
						 int options);

XMLPUBFUN int XMLCALL xmlModuleSymbol		(xmlModulePtr module,
						 const char* name,
						 void **result);

XMLPUBFUN int XMLCALL xmlModuleClose		(xmlModulePtr module);

XMLPUBFUN int XMLCALL xmlModuleFree		(xmlModulePtr module);

#ifdef __cplusplus
}
#endif

#endif /* LIBXML_MODULES_ENABLED */

#endif /*__XML_MODULE_H__ */
