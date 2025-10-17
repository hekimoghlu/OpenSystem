/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 30, 2024.
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

#define IN_LIBEXSLT
#include "libexslt.h"

#include <libxml/tree.h>
#include <libxml/xpath.h>
#include <libxml/xpathInternals.h>

#include "xsltutils.h"
#include "xsltInternals.h"
#include "extensions.h"
#include "transform.h"
#include "extra.h"
#include "preproc.h"

#include "exslt.h"

static void
exsltNodeSetFunction (xmlXPathParserContextPtr ctxt, int nargs) {
    if (nargs != 1) {
	xmlXPathSetArityError(ctxt);
	return;
    }
    if (xmlXPathStackIsNodeSet (ctxt)) {
	xsltFunctionNodeSet (ctxt, nargs);
	return;
    } else {
	xmlDocPtr fragment;
	xsltTransformContextPtr tctxt = xsltXPathGetTransformContext(ctxt);
	xmlNodePtr txt;
	xmlChar *strval;
	xmlXPathObjectPtr obj;
	/*
	* SPEC EXSLT:
	* "You can also use this function to turn a string into a text
	* node, which is helpful if you want to pass a string to a
	* function that only accepts a node-set."
	*/
	fragment = xsltCreateRVT(tctxt);
	if (fragment == NULL) {
	    xsltTransformError(tctxt, NULL, tctxt->inst,
		"exsltNodeSetFunction: Failed to create a tree fragment.\n");
	    tctxt->state = XSLT_STATE_STOPPED;
	    return;
	}
	xsltRegisterLocalRVT(tctxt, fragment);

	strval = xmlXPathPopString (ctxt);

	txt = xmlNewDocText (fragment, strval);
	xmlAddChild((xmlNodePtr) fragment, txt);
	obj = xmlXPathNewNodeSet(txt);
	if (obj == NULL) {
	    xsltTransformError(tctxt, NULL, tctxt->inst,
		"exsltNodeSetFunction: Failed to create a node set object.\n");
	    tctxt->state = XSLT_STATE_STOPPED;
	}
	if (strval != NULL)
	    xmlFree (strval);

	valuePush (ctxt, obj);
    }
}

static void
exsltObjectTypeFunction (xmlXPathParserContextPtr ctxt, int nargs) {
    xmlXPathObjectPtr obj, ret;

    if (nargs != 1) {
	xmlXPathSetArityError(ctxt);
	return;
    }

    obj = valuePop(ctxt);

    switch (obj->type) {
    case XPATH_STRING:
	ret = xmlXPathNewCString("string");
	break;
    case XPATH_NUMBER:
	ret = xmlXPathNewCString("number");
	break;
    case XPATH_BOOLEAN:
	ret = xmlXPathNewCString("boolean");
	break;
    case XPATH_NODESET:
	ret = xmlXPathNewCString("node-set");
	break;
    case XPATH_XSLT_TREE:
	ret = xmlXPathNewCString("RTF");
	break;
    case XPATH_USERS:
	ret = xmlXPathNewCString("external");
	break;
    default:
	xsltGenericError(xsltGenericErrorContext,
		"object-type() invalid arg\n");
	ctxt->error = XPATH_INVALID_TYPE;
	xmlXPathFreeObject(obj);
	return;
    }
    xmlXPathFreeObject(obj);
    valuePush(ctxt, ret);
}


/**
 * exsltCommonRegister:
 *
 * Registers the EXSLT - Common module
 */

void
exsltCommonRegister (void) {
    xsltRegisterExtModuleFunction((const xmlChar *) "node-set",
				  EXSLT_COMMON_NAMESPACE,
				  exsltNodeSetFunction);
    xsltRegisterExtModuleFunction((const xmlChar *) "object-type",
				  EXSLT_COMMON_NAMESPACE,
				  exsltObjectTypeFunction);
    xsltRegisterExtModuleElement((const xmlChar *) "document",
				 EXSLT_COMMON_NAMESPACE,
				 xsltDocumentComp,
				 xsltDocumentElem);
}
