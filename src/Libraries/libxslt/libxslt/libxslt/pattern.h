/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 20, 2024.
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
#ifndef __XML_XSLT_PATTERN_H__
#define __XML_XSLT_PATTERN_H__

#include "xsltInternals.h"
#include "xsltexports.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * xsltCompMatch:
 *
 * Data structure used for the implementation of patterns.
 * It is kept private (in pattern.c).
 */
typedef struct _xsltCompMatch xsltCompMatch;
typedef xsltCompMatch *xsltCompMatchPtr;

/*
 * Pattern related interfaces.
 */

XSLTPUBFUN xsltCompMatchPtr XSLTCALL
		xsltCompilePattern	(const xmlChar *pattern,
					 xmlDocPtr doc,
					 xmlNodePtr node,
					 xsltStylesheetPtr style,
					 xsltTransformContextPtr runtime);
XSLTPUBFUN void XSLTCALL
		xsltFreeCompMatchList	(xsltCompMatchPtr comp);
XSLTPUBFUN int XSLTCALL
		xsltTestCompMatchList	(xsltTransformContextPtr ctxt,
					 xmlNodePtr node,
					 xsltCompMatchPtr comp);
XSLTPUBFUN void XSLTCALL
		xsltCompMatchClearCache	(xsltTransformContextPtr ctxt,
					 xsltCompMatchPtr comp) LIBXSLT_API_AVAILABLE_MACOS13_IOS16_WATCHOS9_TVOS16;
XSLTPUBFUN void XSLTCALL
		xsltNormalizeCompSteps	(void *payload,
					 void *data,
					 const xmlChar *name);

/*
 * Template related interfaces.
 */
XSLTPUBFUN int XSLTCALL
		xsltAddTemplate		(xsltStylesheetPtr style,
					 xsltTemplatePtr cur,
					 const xmlChar *mode,
					 const xmlChar *modeURI);
XSLTPUBFUN xsltTemplatePtr XSLTCALL
		xsltGetTemplate		(xsltTransformContextPtr ctxt,
					 xmlNodePtr node,
					 xsltStylesheetPtr style);
XSLTPUBFUN void XSLTCALL
		xsltFreeTemplateHashes	(xsltStylesheetPtr style);
XSLTPUBFUN void XSLTCALL
		xsltCleanupTemplates	(xsltStylesheetPtr style);

#if 0
int		xsltMatchPattern	(xsltTransformContextPtr ctxt,
					 xmlNodePtr node,
					 const xmlChar *pattern,
					 xmlDocPtr ctxtdoc,
					 xmlNodePtr ctxtnode);
#endif
#ifdef __cplusplus
}
#endif

#endif /* __XML_XSLT_PATTERN_H__ */

