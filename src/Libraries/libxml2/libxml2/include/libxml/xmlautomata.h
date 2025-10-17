/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 23, 2022.
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
#ifndef __XML_AUTOMATA_H__
#define __XML_AUTOMATA_H__

#include <libxml/xmlversion.h>
#include <libxml/tree.h>

#ifdef LIBXML_REGEXP_ENABLED
#ifdef LIBXML_AUTOMATA_ENABLED
#include <libxml/xmlregexp.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * xmlAutomataPtr:
 *
 * A libxml automata description, It can be compiled into a regexp
 */
typedef struct _xmlAutomata xmlAutomata;
typedef xmlAutomata *xmlAutomataPtr;

/**
 * xmlAutomataStatePtr:
 *
 * A state int the automata description,
 */
typedef struct _xmlAutomataState xmlAutomataState;
typedef xmlAutomataState *xmlAutomataStatePtr;

/*
 * Building API
 */
XMLPUBFUN xmlAutomataPtr XMLCALL
		    xmlNewAutomata		(void);
XMLPUBFUN void XMLCALL
		    xmlFreeAutomata		(xmlAutomataPtr am);

XMLPUBFUN xmlAutomataStatePtr XMLCALL
		    xmlAutomataGetInitState	(xmlAutomataPtr am);
XMLPUBFUN int XMLCALL
		    xmlAutomataSetFinalState	(xmlAutomataPtr am,
						 xmlAutomataStatePtr state);
XMLPUBFUN xmlAutomataStatePtr XMLCALL
		    xmlAutomataNewState		(xmlAutomataPtr am);
XMLPUBFUN xmlAutomataStatePtr XMLCALL
		    xmlAutomataNewTransition	(xmlAutomataPtr am,
						 xmlAutomataStatePtr from,
						 xmlAutomataStatePtr to,
						 const xmlChar *token,
						 void *data);
XMLPUBFUN xmlAutomataStatePtr XMLCALL
		    xmlAutomataNewTransition2	(xmlAutomataPtr am,
						 xmlAutomataStatePtr from,
						 xmlAutomataStatePtr to,
						 const xmlChar *token,
						 const xmlChar *token2,
						 void *data);
XMLPUBFUN xmlAutomataStatePtr XMLCALL
                    xmlAutomataNewNegTrans	(xmlAutomataPtr am,
						 xmlAutomataStatePtr from,
						 xmlAutomataStatePtr to,
						 const xmlChar *token,
						 const xmlChar *token2,
						 void *data);

XMLPUBFUN xmlAutomataStatePtr XMLCALL
		    xmlAutomataNewCountTrans	(xmlAutomataPtr am,
						 xmlAutomataStatePtr from,
						 xmlAutomataStatePtr to,
						 const xmlChar *token,
						 int min,
						 int max,
						 void *data);
XMLPUBFUN xmlAutomataStatePtr XMLCALL
		    xmlAutomataNewCountTrans2	(xmlAutomataPtr am,
						 xmlAutomataStatePtr from,
						 xmlAutomataStatePtr to,
						 const xmlChar *token,
						 const xmlChar *token2,
						 int min,
						 int max,
						 void *data);
XMLPUBFUN xmlAutomataStatePtr XMLCALL
		    xmlAutomataNewOnceTrans	(xmlAutomataPtr am,
						 xmlAutomataStatePtr from,
						 xmlAutomataStatePtr to,
						 const xmlChar *token,
						 int min,
						 int max,
						 void *data);
XMLPUBFUN xmlAutomataStatePtr XMLCALL
		    xmlAutomataNewOnceTrans2	(xmlAutomataPtr am,
						 xmlAutomataStatePtr from,
						 xmlAutomataStatePtr to,
						 const xmlChar *token,
						 const xmlChar *token2,
						 int min,
						 int max,
						 void *data);
XMLPUBFUN xmlAutomataStatePtr XMLCALL
		    xmlAutomataNewAllTrans	(xmlAutomataPtr am,
						 xmlAutomataStatePtr from,
						 xmlAutomataStatePtr to,
						 int lax);
XMLPUBFUN xmlAutomataStatePtr XMLCALL
		    xmlAutomataNewEpsilon	(xmlAutomataPtr am,
						 xmlAutomataStatePtr from,
						 xmlAutomataStatePtr to);
XMLPUBFUN xmlAutomataStatePtr XMLCALL
		    xmlAutomataNewCountedTrans	(xmlAutomataPtr am,
						 xmlAutomataStatePtr from,
						 xmlAutomataStatePtr to,
						 int counter);
XMLPUBFUN xmlAutomataStatePtr XMLCALL
		    xmlAutomataNewCounterTrans	(xmlAutomataPtr am,
						 xmlAutomataStatePtr from,
						 xmlAutomataStatePtr to,
						 int counter);
XMLPUBFUN int XMLCALL
		    xmlAutomataNewCounter	(xmlAutomataPtr am,
						 int min,
						 int max);

XMLPUBFUN xmlRegexpPtr XMLCALL
		    xmlAutomataCompile		(xmlAutomataPtr am);
XMLPUBFUN int XMLCALL
		    xmlAutomataIsDeterminist	(xmlAutomataPtr am);

#ifdef __cplusplus
}
#endif

#endif /* LIBXML_AUTOMATA_ENABLED */
#endif /* LIBXML_REGEXP_ENABLED */

#endif /* __XML_AUTOMATA_H__ */
