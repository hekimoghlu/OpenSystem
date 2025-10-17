/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 9, 2023.
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
#ifndef __XML_XSLT_EXTRA_H__
#define __XML_XSLT_EXTRA_H__

#include <libxml/xpath.h>
#include "xsltexports.h"
#include "xsltInternals.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * XSLT_LIBXSLT_NAMESPACE:
 *
 * This is the libxslt namespace for specific extensions.
 */
#define XSLT_LIBXSLT_NAMESPACE ((xmlChar *) "http://xmlsoft.org/XSLT/namespace")

/**
 * XSLT_SAXON_NAMESPACE:
 *
 * This is Michael Kay's Saxon processor namespace for extensions.
 */
#define XSLT_SAXON_NAMESPACE ((xmlChar *) "http://icl.com/saxon")

/**
 * XSLT_XT_NAMESPACE:
 *
 * This is James Clark's XT processor namespace for extensions.
 */
#define XSLT_XT_NAMESPACE ((xmlChar *) "http://www.jclark.com/xt")

/**
 * XSLT_XALAN_NAMESPACE:
 *
 * This is the Apache project XALAN processor namespace for extensions.
 */
#define XSLT_XALAN_NAMESPACE ((xmlChar *)	\
	                        "org.apache.xalan.xslt.extensions.Redirect")


XSLTPUBFUN void XSLTCALL
		xsltFunctionNodeSet	(xmlXPathParserContextPtr ctxt,
					 int nargs);
XSLTPUBFUN void XSLTCALL
		xsltDebug		(xsltTransformContextPtr ctxt,
					 xmlNodePtr node,
					 xmlNodePtr inst,
					 xsltElemPreCompPtr comp);


XSLTPUBFUN void XSLTCALL
		xsltRegisterExtras	(xsltTransformContextPtr ctxt);
XSLTPUBFUN void XSLTCALL
		xsltRegisterAllExtras	(void);

#ifdef __cplusplus
}
#endif

#endif /* __XML_XSLT_EXTRA_H__ */

