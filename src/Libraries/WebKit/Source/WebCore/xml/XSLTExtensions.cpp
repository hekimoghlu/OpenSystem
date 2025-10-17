/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 6, 2025.
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
#include "config.h"

#if ENABLE(XSLT)
#include "XSLTExtensions.h"

// FIXME (286277): Stop ignoring -Wundef and -Wdeprecated-declarations in code that imports libxml and libxslt headers
IGNORE_WARNINGS_BEGIN("deprecated-declarations")
IGNORE_WARNINGS_BEGIN("undef")
#include <libxml/xpathInternals.h>

#include <libxslt/extensions.h>
#include <libxslt/extra.h>
#include <libxslt/xsltutils.h>
IGNORE_WARNINGS_END
IGNORE_WARNINGS_END

namespace WebCore {

// FIXME: This code is taken from libexslt v1.1.35; should sync with newer versions.
static void exsltNodeSetFunction(xmlXPathParserContextPtr ctxt, int nargs)
{
    xmlDocPtr fragment;
    xsltTransformContextPtr tctxt = xsltXPathGetTransformContext(ctxt);
    xmlNodePtr txt;
    xmlChar *strval;
    xmlXPathObjectPtr obj;

    if (nargs != 1) {
        xmlXPathSetArityError(ctxt);
        return;
    }

    if (xmlXPathStackIsNodeSet(ctxt)) {
        xsltFunctionNodeSet(ctxt, nargs);
        return;
    }

    /*
     * SPEC EXSLT:
     * "You can also use this function to turn a string into a text
     * node, which is helpful if you want to pass a string to a
     * function that only accepts a node-set."
     */
    fragment = xsltCreateRVT(tctxt);
    if (!fragment) {
        xsltTransformError(tctxt, nullptr, tctxt->inst,
            "WebCore::exsltNodeSetFunction: Failed to create a tree fragment.\n");
        tctxt->state = XSLT_STATE_STOPPED;
        return;
    }
    xsltRegisterLocalRVT(tctxt, fragment);

    strval = xmlXPathPopString(ctxt);

    txt = xmlNewDocText(fragment, strval);
    xmlAddChild(reinterpret_cast<xmlNodePtr>(fragment), txt);
    obj = xmlXPathNewNodeSet(txt);

    // FIXME: It might be helpful to push any errors from xmlXPathNewNodeSet
    // up to the Javascript Console.
    if (!obj) {
        xsltTransformError(tctxt, nullptr, tctxt->inst,
            "WebCore::exsltNodeSetFunction: Failed to create a node set object.\n");
        tctxt->state = XSLT_STATE_STOPPED;
    }
// FIXME (286277): Stop ignoring -Wundef and -Wdeprecated-declarations in code that imports libxml and libxslt headers
IGNORE_WARNINGS_BEGIN("deprecated-declarations")
    if (strval != NULL)
        xmlFree(strval);
IGNORE_WARNINGS_END

    valuePush(ctxt, obj);
}

void registerXSLTExtensions(xsltTransformContextPtr ctxt)
{
    xsltRegisterExtFunction(ctxt, (const xmlChar*)"node-set", (const xmlChar*)"http://exslt.org/common", exsltNodeSetFunction);
}

}

#endif
