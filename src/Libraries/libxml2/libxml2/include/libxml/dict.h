/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 21, 2023.
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
#ifndef __XML_DICT_H__
#define __XML_DICT_H__

#include <stddef.h>
#include <libxml/xmlstring.h>
#include <libxml/xmlversion.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * The dictionary.
 */
typedef struct _xmlDict xmlDict;
typedef xmlDict *xmlDictPtr;

/*
 * Initializer
 */
XMLPUBFUN int XMLCALL  xmlInitializeDict(void);

/*
 * Constructor and destructor.
 */
XMLPUBFUN xmlDictPtr XMLCALL
			xmlDictCreate	(void);
XMLPUBFUN size_t XMLCALL
			xmlDictSetLimit	(xmlDictPtr dict,
                                         size_t limit);
XMLPUBFUN size_t XMLCALL
			xmlDictGetUsage (xmlDictPtr dict);
XMLPUBFUN xmlDictPtr XMLCALL
			xmlDictCreateSub(xmlDictPtr sub);
XMLPUBFUN int XMLCALL
			xmlDictReference(xmlDictPtr dict);
XMLPUBFUN void XMLCALL
			xmlDictFree	(xmlDictPtr dict);

/*
 * Lookup of entry in the dictionary.
 */
XMLPUBFUN const xmlChar * XMLCALL
			xmlDictLookup	(xmlDictPtr dict,
		                         const xmlChar *name,
		                         int len);
XMLPUBFUN const xmlChar * XMLCALL
			xmlDictExists	(xmlDictPtr dict,
		                         const xmlChar *name,
		                         int len);
XMLPUBFUN const xmlChar * XMLCALL
			xmlDictQLookup	(xmlDictPtr dict,
		                         const xmlChar *prefix,
		                         const xmlChar *name);
XMLPUBFUN int XMLCALL
			xmlDictOwns	(xmlDictPtr dict,
					 const xmlChar *str);
XMLPUBFUN int XMLCALL
			xmlDictSize	(xmlDictPtr dict);

/*
 * Cleanup function
 */
XMLPUBFUN void XMLCALL
                        xmlDictCleanup  (void);

#ifdef __cplusplus
}
#endif
#endif /* ! __XML_DICT_H__ */
