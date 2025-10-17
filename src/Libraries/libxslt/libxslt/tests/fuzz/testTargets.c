/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 23, 2021.
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

#include "fuzz.h"
#include <libxml/globals.h>

int
testXPath(int argc, char **argv) {
    xmlXPathObjectPtr obj;
    const char expr[] = "count(//node())";
    int ret = 0;

    if (xsltFuzzXPathInit(&argc, &argv, argv[1]) != 0) {
        xsltFuzzXPathCleanup();
        return 1;
    }

    obj = xsltFuzzXPath(expr, sizeof(expr) - 1);
    if ((obj == NULL) || (obj->type != XPATH_NUMBER)) {
        fprintf(stderr, "Expression doesn't evaluate to number\n");
        ret = 1;
    } else if (obj->floatval != 39.0) {
        fprintf(stderr, "Expression returned %f, expected %f\n",
                obj->floatval, 39.0);
        ret = 1;
    }

    xsltFuzzXPathFreeObject(obj);
    xsltFuzzXPathCleanup();

    return ret;
}

int
testXslt(int argc, char **argv) {
    xmlChar *result;
    const char styleBuf[] =
        "<xsl:stylesheet"
        " xmlns:xsl='http://www.w3.org/1999/XSL/Transform'"
        " version='1.0'"
        " extension-element-prefixes='"
        "  exsl exslt crypto date dyn math set str saxon"
        "'>\n"
        "<xsl:output omit-xml-declaration='yes'/>\n"
        "<xsl:template match='/'>\n"
        " <r><xsl:value-of select='count(//node())'/></r>\n"
        "</xsl:template>\n"
        "</xsl:stylesheet>\n";
    int ret = 0;

    if (xsltFuzzXsltInit(&argc, &argv, argv[1]) != 0) {
        xsltFuzzXsltCleanup();
        return 1;
    }

    result = xsltFuzzXslt(styleBuf, sizeof(styleBuf) - 1);
    if (result == NULL) {
        fprintf(stderr, "Result is NULL\n");
        ret = 1;
    } else if (xmlStrcmp(result, BAD_CAST "<r>42</r>\n") != 0) {
        fprintf(stderr, "Stylesheet returned\n%sexpected \n%s\n",
                result, "<r>42</r>");
        ret = 1;
    }

    xmlFree(result);
    xsltFuzzXsltCleanup();

    return ret;
}

int main(int argc, char **argv) {
    int ret = 0;

    if (testXPath(argc, argv) != 0)
        ret = 1;
    if (testXslt(argc, argv) != 0)
        ret = 1;

    return ret;
}
